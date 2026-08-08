"""
PHASE 41 (T6.4) -- POLYSEMY: ACT ON DETECTION. Close the loop from the
detector (phases 12/21/27/28) to the field mechanism (phase 10), then score
DOWNSTREAM UTILITY AT MATCHED CAPACITY.

The project detects sense structure from corpus statistics and, per the
README thread, "now needs the field mechanism to act on it". This phase
wires detection to splitting end-to-end and asks the only question that
converts a detector into a capability: does the split BUY anything?

  detect  -- per-word predictive gain (phase 12's Myhill-Nerode statistic)
             over UNSUPERVISED categories (discover_categories_v2, the
             standing method since phase 21), each word tested against ITS
             OWN permutation null at ITS OWN sample size (phase 21/27's
             committed operationalization: "clears its own p99 null").
  split   -- detected words, and only detected words, are allowed to recruit
             a second sense-slot under phase 10's context-primed settling.
  score   -- successor distinctness vs a same-word permutation null, plus
             held-out next-category / next-word prediction and generation,
             against a MATCHED-CAPACITY control.

WHAT THIS PHASE INHERITS AND MUST NOT RE-LITIGATE
-------------------------------------------------
1. Phase 10's TWICE-CONFIRMED NEGATIVE: attractor pull during perception
   (g_mem > 0) suppresses splitting in both orderings -- pull-before-gate
   erases the context evidence (3/3 -> 0/3 role coverage), pull-after-gate
   collapses to one slot per dual word. PERCEPTION READS THE FIELD, IT DOES
   NOT BEND IT. Every arm here runs at g_mem = 0.0. No arm re-tests it.
2. T1.5 / phase 33d: EXTRA SLOTS ALONE BUY ACCURACY (ACC rose monotonically
   0.712 -> 0.900 on K alone). Any split-vs-unsplit comparison is therefore
   confounded by capacity, and a matched-capacity control is MANDATORY, not
   optional. The `matched` arm below is that control.
3. Phase 28: the gain statistic conflates lexical polysemy with grammatical
   context-sensitivity on real text. This phase runs on the phase-8/10
   synthetic corpus, where the dual words (fish/duck/bear) are genuinely
   LEXICAL -- same surface form, same embedding, two grammatical roles --
   so the conflation is not in play and the mechanism question is clean.
   That is a scope choice, stated here, not a claim about real text.

ARMS (identical class, identical dynamics, identical routing; the ONLY
difference is which words may split and what decides where an occurrence
goes). All at g_mem = 0.0, phase 10's committed thr = 0.65 / merge = 0.45.

  unsplit   -- no word may split. One slot per word. The capability floor.
  split     -- detected words may split, occurrences routed by the
               context-primed dynamics (phase 10's own assignment rule).
  matched   -- THE MANDATORY CONTROL. Constructed to have EXACTLY the split
               arm's per-word slot count for EVERY word (hence exactly its
               live-slot total), but occurrences are routed to sense-slots
               AT RANDOM -- context-blind. Same learning rule, same slot
               budget, same per-word structure; only the information driving
               the partition differs. If `split` cannot beat `matched`, the
               split is carrying no usable information.
  randword  -- diagnostic: |D| RANDOMLY CHOSEN UNDETECTED words are allowed
               to split under the same context-primed dynamics. Tests
               whether the DETECTOR (not merely splitting) is load-bearing.
               Its slot count is reported, not matched.

Why `matched` skips consolidation, stated plainly: phase 10's consolidate
merges slot pairs whose context residuals agree, i.e. it is a duplicate
detector. Applied to a RANDOM partition it would by construction undo that
partition and hand the control LESS capacity than the arm it controls for --
biasing the comparison toward `split`. The control is therefore built
directly at the split arm's post-consolidate per-word structure and is not
consolidated again. Live-slot totals are asserted equal and printed.

PRE-REGISTERED PREDICTIONS (frozen before the run; committed in the
pre-registration commit, results appended afterward)
----------------------------------------------------------------------
(a) SUCCESSOR DISTINCTNESS. Sense-split slots show distinct successor
    distributions vs a SAME-WORD permutation null (occurrences re-partitioned
    at the observed group sizes, 200 draws, p99). Operationalized: for each
    split word, mean pairwise total-variation distance between its slots'
    successor-category distributions exceeds that word's own p99. Success =
    3/3 dual words clear it.
(b) DOWNSTREAM UTILITY AT MATCHED CAPACITY. `split` beats `matched` on
    held-out next-category accuracy and/or next-word accuracy. SURVIVAL RULE,
    taken verbatim from T1.6's (the rule this project already pays for):
    mean paired delta > +0.02 AND NO SIGN FLIP across seeds s = 0..4.
    A metric that fails either clause does not survive.
(c) HONEST NEGATIVE. If utility is FLAT at matched capacity, sense splitting
    is a representational nicety and not a functional gain -- and that is a
    publishable negative about the whole polysemy line, to be reported
    plainly rather than buried under the split-vs-unsplit delta (which T1.5
    already told us is capacity).

NO GRID IS SEARCHED. Every mechanism constant is taken from a committed
phase: thr=0.65 / g_mem=0.0 / resid_merge=0.45 / word_thresh=0.7 / hold=12
from phase 10's seed-stable best config; recruit=0.5 / merge_thresh=0.84 /
prune_frac=0.02 from phase 8/15's baseline recipe; p99 per-word nulls from
phase 21/27; the +0.02 / no-sign-flip survival rule from T1.6. Because
nothing is grid-selected, the T1.8/T1.9 held-out-seed requirement is not
triggered -- but every arm is still reseeded s = 0..4 and paired, per the
Category 6 standing rule, and the baseline is reseeded on the same split.

Seeds: s in 0..4 -> organism seed s, train stream seed 99+s, held-out eval
stream seed 900+s. s=0 is phase 10's exact stream (seed 99), so the s=0 row
is the reproduction anchor for phase 10's corpus.

Run:  python phase41_detection_driven_split.py
      PHASE41_SEEDS=0,1  python phase41_detection_driven_split.py   (subset)
"""

import os
import sys
import time
from collections import Counter

import numpy as np

# phase 10's module runs a sweep at import time under its default STAGE;
# this neutralizes that (any unrecognized STAGE is a no-op) so we can reuse
# its corpus, its PrimedOrg dynamics and its helpers without re-running it.
os.environ.setdefault('PHASE10_STAGE', 'phase41-import-only')

import phase10_context_primed_settling as p10          # noqa: E402
from organism import normalize                          # noqa: E402
from polysemy_organism import PolysemyOrganism          # noqa: E402

# ---- corpus constants, taken from phase 10 verbatim ---------------------------
N = p10.N
NORM = np.sqrt(N)
EMB = p10.embeddings
VOCAB = p10.vocab
N_WORDS = p10.N_WORDS
DUAL_WORDS = p10.DUAL_WORDS
DUAL_IDX = {p10.word_to_idx[w] for w in DUAL_WORDS}
NEXT_CAT = p10.NEXT_CAT
CAT_NAMES = p10.cat_names
SEQ_LEN = p10.SEQ_LEN            # 12000, phase 10's training length
EVAL_LEN = 4000                  # held-out, never perceived
P_DUAL = 0.55                    # phase 10's committed stream parameter

# ---- mechanism constants, all from committed phases ---------------------------
K_SLOTS = 70                     # phase 10 / phase 8
HOLD = 12
G_IN = 5.0
DT = 0.05
ETA = 0.2                        # phase 10's PrimedOrg default
G_MEM = 0.0                      # phase 10's twice-confirmed negative: NO pull
RESID_RECRUIT = 0.65             # phase 10's seed-stable best config
RESID_MERGE = 0.45
WORD_THRESH = 0.7
BASE_RECRUIT = 0.5               # phase 8/15 baseline perceive
BASE_MERGE = 0.84
BASE_PRUNE = 0.02
NULL_DRAWS = 200                 # per-word permutation nulls
NULL_PCT = 99                    # phase 21/27's "clears its own p99 null"
SURVIVE_DELTA = 0.02             # T1.6's survival threshold
GEN_STEPS = 1000
TAU_H = 15.0
LAM_H = 2.5

BLOCK_SPLIT = 2.0                # novelty <= 1 always, so this gate never fires


# ============================================================================
# helpers
# ============================================================================
def entropy(counts):
    c = np.asarray(list(counts), dtype=float)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c[c > 0] / tot
    return float(-np.sum(p * np.log2(p)))


def cond_gain(pred, succ):
    """Phase 12's statistic: H(succ) - H(succ | pred), observable quantities."""
    n = len(pred)
    if n == 0:
        return 0.0
    h_u = entropy(Counter(succ.tolist()).values())
    h_c = 0.0
    for pc in set(pred.tolist()):
        sub = succ[pred == pc]
        h_c += (len(sub) / n) * entropy(Counter(sub.tolist()).values())
    return float(h_u - h_c)


def plain_stream(wseq, hold=HOLD):
    for w in wseq:
        s = normalize(EMB[w].astype(complex), NORM)
        for _ in range(hold):
            yield s


def tv_distance(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def dist_over(labels, n_lab):
    c = np.bincount(np.asarray(labels, dtype=int), minlength=n_lab).astype(float)
    return c / max(c.sum(), 1.0)


# ============================================================================
# Stage B -- detection (label-free end to end)
# ============================================================================
def detect(train_seq, seed):
    """Baseline organism -> unsupervised categories -> per-word predictive
    gain -> per-word permutation null. Returns everything downstream needs.
    No ground-truth role is touched anywhere in this function."""
    org = PolysemyOrganism(N=N, K=K_SLOTS, omega=0.15, beta=10.0, seed=seed,
                           backend="auto")
    org.perceive(list(plain_stream(train_seq)), g_in=G_IN, dt=DT, eta=0.02,
                 recruit=BASE_RECRUIT)
    org.consolidate(merge_thresh=BASE_MERGE, prune_frac=BASE_PRUNE)

    states = np.array([normalize(EMB[w].astype(complex), NORM) for w in train_seq])
    asg = np.abs((org.mem.conj() @ states.T) / N).argmax(0)
    slot_word = {}
    for k in range(org.mem.shape[0]):
        members = np.array(train_seq)[asg == k]
        if len(members):
            slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
    coverage = len(set(slot_word.values()))

    # k_range starts at 2 so a genuine 2-way collapse is expressible (phase 27
    # measured exactly that failure mode at 5M words); the upper end is well
    # above the corpus's 3 categories.
    cat_res = org.discover_categories_v2(k_range=range(2, 9), seed=3)
    word_cat = {}
    for k, w in slot_word.items():
        word_cat[w] = org.word_slot_to_cat.get(k)

    cats = np.array([word_cat.get(w, -1) for w in train_seq])
    rng = np.random.default_rng(10_000 + seed)
    rows = []
    for wi in range(N_WORDS):
        occ = np.array([t for t, w in enumerate(train_seq)
                        if w == wi and 0 < t < len(train_seq) - 1])
        if len(occ) < 30:
            continue
        pred, succ = cats[occ - 1], cats[occ + 1]
        keep = (pred >= 0) & (succ >= 0)
        pred, succ = pred[keep], succ[keep]
        if len(pred) < 30:
            continue
        g = cond_gain(pred, succ)
        # null: shuffle the CONDITIONING variable within this word's own
        # occurrences -- destroys the prev->succ association, preserves both
        # marginals and the sample size. Phase 21/27's per-word null.
        null = np.array([cond_gain(rng.permutation(pred), succ)
                         for _ in range(NULL_DRAWS)])
        p99 = float(np.percentile(null, NULL_PCT))
        rows.append(dict(word=wi, n=len(pred), gain=g, null=p99,
                         margin=g - p99, detected=g > p99))
    detected = {r['word'] for r in rows if r['detected']}
    return dict(org=org, slot_word=slot_word, coverage=coverage,
                n_categories=cat_res['n_categories'],
                silhouette=cat_res['silhouette'], word_cat=word_cat,
                gain_rows=rows, detected=detected)


# ============================================================================
# Stage C -- gated context-primed splitting (phase 10's dynamics, gated)
# ============================================================================
class GatedPrimedOrg(p10.PrimedOrg):
    """Phase 10's PrimedOrg with two additions and no dynamical changes:

    * `allowed`: only words in this set may recruit a SECOND slot. Undetected
      words collapse context into one slot -- phase 13/16's Myhill-Nerode
      discipline ("if context doesn't change what comes next, merge the
      contexts"), now driven by a per-word measured null instead of a
      threshold.
    * `forced`: word_id -> per-occurrence group index. When given, that
      word's occurrences are routed to its group's slot directly, bypassing
      the residual gate. This is how the matched-capacity control is built:
      identical slot budget and per-word structure, context-blind routing.

    The settling dynamics, the residual gate, the update rule, the transition
    bookkeeping and consolidate() are phase 10's, untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_slots = {}       # observational: word_id -> [slot ids]
        self.group_slot = {}       # (word_id, group_id) -> slot id, forced mode

    def perceive_gated(self, seq, allowed=frozenset(), forced=None,
                       hold=HOLD, g_in=G_IN, g_mem=G_MEM, dt=DT, eta=ETA,
                       word_thresh=WORD_THRESH, resid_recruit=RESID_RECRUIT):
        forced = forced or {}
        seen = {w: 0 for w in forced}
        assigns = np.full(len(seq), -1, dtype=int)
        self.composites = np.zeros((len(seq), self.N), dtype=complex)
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)

        for t, wi in enumerate(seq):
            x = normalize(EMB[wi].astype(complex), self.norm)
            # frame 1: input-only settling. The carry-over context must be
            # OBSERVED before anything acts on it (phase 10's negative).
            z = self._settle(z, x, g_in, 0.0, dt)
            v = z.copy()
            self.composites[t] = v

            k = -1
            gid = None
            if wi in forced:
                gid = int(forced[wi][seen[wi]])
                seen[wi] += 1
                k = self.group_slot.get((wi, gid), -1)
            else:
                # phase 10's candidate rule, verbatim: same-word candidates by
                # word-direction overlap, then the residual novelty gate. The
                # gate is unreachable for a word outside `allowed` (novelty is
                # bounded by 1 and the blocked threshold is 2.0).
                used_idx = np.where(self.used)[0]
                thr = resid_recruit if wi in allowed else BLOCK_SPLIT
                if len(used_idx) > 0:
                    wov = np.abs(self.overlaps(x, self.wvec[used_idx]))
                    cands = used_idx[wov > word_thresh]
                    if len(cands) > 0:
                        r_in = p10.ctx_residual(v, x)
                        rovs = [p10.resid_overlap(r_in, self.slot_residual(c))
                                for c in cands]
                        if (1 - max(rovs)) > thr and not self.used.all():
                            k = -1                     # same word, novel context
                        else:
                            zb = (self._settle(v, x, g_in, g_mem, dt, steps=4)
                                  if g_mem > 0 else v)
                            cov = np.abs(self.overlaps(zb, self.xi[cands]))
                            k = int(cands[int(np.argmax(cov))])
                            z = zb

            if k < 0:
                if not self.used.all():
                    k = int(np.argmin(self.used.astype(float)))
                    self.xi[k] = v
                    self.wvec[k] = x
                    self.used[k] = True
                    self.word_slots.setdefault(wi, []).append(k)
                else:
                    used_idx = np.where(self.used)[0]
                    ovs = np.abs(self.overlaps(v, self.xi[used_idx]))
                    k = int(used_idx[np.argmax(ovs)])
                if gid is not None:
                    self.group_slot[(wi, gid)] = k
            else:
                ph = np.exp(-1j * np.angle(self.overlaps(v, self.xi[[k]])[0]))
                self.xi[k] = normalize(self.xi[k] + eta * (v * ph - self.xi[k]),
                                       self.norm)
                if k not in self.word_slots.setdefault(wi, []):
                    self.word_slots[wi].append(k)
            self.count[k] += 1
            z = self._finish(t, k, z, x, g_in, dt, hold, eta, assigns)
        return assigns

    def _finish(self, t, k, z, x, g_in, dt, hold, eta, assigns):
        """frames 2..hold: input-only refinement toward the pure word. Returns
        the settled state, which seeds the NEXT word's context carry-over --
        that carry-over IS the context signal (phase 10), so it must flow back
        into the stream loop."""
        for _ in range(hold - 1):
            z = self._settle(z, x, g_in, 0.0, dt)
        pw = np.exp(-1j * np.angle(self.overlaps(z, self.wvec[[k]])[0]))
        self.wvec[k] = normalize(self.wvec[k] + eta * (z * pw - self.wvec[k]),
                                 self.norm)
        if self.prev_slot >= 0:
            self.Pn[self.prev_slot, k] += 1
        self.prev_slot = k
        assigns[t] = k
        return z

    # ---- frozen routing on a held-out stream (no learning, no recruiting) ----
    def route(self, seq, rng_seed, hold=HOLD, g_in=G_IN, dt=DT,
              word_thresh=WORD_THRESH):
        """Phase 10's own assignment rule at g_mem=0, applied frozen: settle
        frame 1 with carry-over, take the same-word candidates by word-direction
        overlap, pick argmax overlap of the context-rich state against their
        stored xi. Identical code and identical RNG for every arm."""
        out = np.full(len(seq), -1, dtype=int)
        rng = np.random.default_rng(rng_seed)
        z = normalize(rng.standard_normal(self.N).astype(complex), self.norm)
        used_idx = np.where(self.used)[0]
        for t, wi in enumerate(seq):
            x = normalize(EMB[wi].astype(complex), self.norm)
            z = self._settle(z, x, g_in, 0.0, dt)
            v = z.copy()
            if len(used_idx):
                wov = np.abs(self.overlaps(x, self.wvec[used_idx]))
                cands = used_idx[wov > word_thresh]
                if len(cands) == 0:
                    cands = used_idx
                cov = np.abs(self.overlaps(v, self.xi[cands]))
                out[t] = int(cands[int(np.argmax(cov))])
            for _ in range(hold - 1):
                z = self._settle(z, x, g_in, 0.0, dt)
        return out


def run_arm(train_seq, seed, allowed=frozenset(), forced=None, consolidate=True):
    org = GatedPrimedOrg(N=N, K=K_SLOTS, seed=seed)
    a = org.perceive_gated(train_seq, allowed=allowed, forced=forced)
    if consolidate:
        a = org.consolidate(a, resid_merge=RESID_MERGE)
    return org, a


def per_word_slot_occupancy(train_seq, assigns, org):
    """word_id -> list of occurrence counts, one per live slot of that word."""
    occ = {}
    for t, w in enumerate(train_seq):
        k = int(assigns[t])
        if k >= 0 and org.used[k]:
            occ.setdefault(w, Counter())[k] += 1
    return {w: sorted(c.values(), reverse=True) for w, c in occ.items()}


def build_forced(train_seq, occupancy, seed):
    """Random per-occurrence group assignment reproducing, for EVERY word, the
    split arm's slot COUNT and (in proportion) its occupancy. Capacity and
    per-word structure match exactly; the routing information does not."""
    rng = np.random.default_rng(50_000 + seed)
    counts = Counter(train_seq)
    forced = {}
    for w, sizes in occupancy.items():
        m = len(sizes)
        n = counts[w]
        if m <= 0 or n <= 0:
            continue
        frac = np.array(sizes, dtype=float)
        frac = frac / frac.sum()
        alloc = np.floor(frac * n).astype(int)
        alloc[0] += n - alloc.sum()
        alloc = np.maximum(alloc, 0)
        if alloc.sum() != n:                     # defensive; keeps totals exact
            alloc[0] += n - alloc.sum()
        groups = np.concatenate([np.full(a, i) for i, a in enumerate(alloc)])
        rng.shuffle(groups)
        forced[w] = groups
    return forced


# ============================================================================
# Stage D -- scoring
# ============================================================================
def slot_maps(train_seq, assigns, org, word_cat, true_roles):
    """slot -> word (observable), slot -> emergent category (observable),
    slot -> dominant TRUE role (EVAL ONLY, never fed back)."""
    s_words, s_roles = {}, {}
    for t, w in enumerate(train_seq):
        k = int(assigns[t])
        if k >= 0 and org.used[k]:
            s_words.setdefault(k, Counter())[w] += 1
            s_roles.setdefault(k, Counter())[true_roles[t]] += 1
    slot_word = {k: c.most_common(1)[0][0] for k, c in s_words.items()}
    slot_role = {k: c.most_common(1)[0][0] for k, c in s_roles.items()}
    slot_cat = {k: word_cat.get(w, -1) for k, w in slot_word.items()}
    return slot_word, slot_cat, slot_role


def successor_distinctness(train_seq, assigns, org, word_cat, detected, seed):
    """Prediction (a): do a word's sense-slots have distinct successor-category
    distributions vs a SAME-WORD permutation null at the observed group sizes?"""
    n_cat = max([c for c in word_cat.values() if c is not None] + [0]) + 1
    succ_cat = np.array([word_cat.get(w, -1) for w in train_seq])
    rng = np.random.default_rng(70_000 + seed)
    out = []
    by_word = {}
    for t in range(len(train_seq) - 1):
        k = int(assigns[t])
        if k < 0 or not org.used[k] or succ_cat[t + 1] < 0:
            continue
        by_word.setdefault(train_seq[t], []).append((k, int(succ_cat[t + 1])))
    for w in sorted(detected):
        pairs = by_word.get(w, [])
        slots = sorted({k for k, _ in pairs})
        if len(slots) < 2 or len(pairs) < 30:
            out.append(dict(word=w, n_slots=len(slots), stat=0.0, null=0.0,
                            distinct=False))
            continue
        labels = np.array([s for s, _ in pairs])
        succs = np.array([c for _, c in pairs])

        def stat(lab):
            ds = [dist_over(succs[lab == s], n_cat) for s in np.unique(lab)]
            pw = [tv_distance(ds[i], ds[j])
                  for i in range(len(ds)) for j in range(i + 1, len(ds))]
            return float(np.mean(pw)) if pw else 0.0

        obs = stat(labels)
        null = np.array([stat(rng.permutation(labels)) for _ in range(NULL_DRAWS)])
        p99 = float(np.percentile(null, NULL_PCT))
        out.append(dict(word=w, n_slots=len(slots), stat=obs, null=p99,
                        distinct=obs > p99))
    return out


def predict_scores(org, routed, eval_seq, slot_word, slot_cat, word_cat):
    """Held-out next-category / next-word prediction from each routed slot's
    learned transition row. Targets are observable (surface word, emergent
    category) -- no ground-truth role anywhere."""
    live = np.where(org.used)[0]
    n_cat = max([c for c in word_cat.values() if c is not None] + [0]) + 1
    cat_of = np.full(org.K, -1)
    word_of = np.full(org.K, -1)
    for k in live:
        cat_of[k] = slot_cat.get(int(k), -1)
        word_of[k] = slot_word.get(int(k), -1)

    rows = org.Pn[np.ix_(live, live)].astype(float)
    cat_mass = np.zeros((len(live), n_cat))
    word_mass = np.zeros((len(live), N_WORDS))
    for j, kj in enumerate(live):
        if cat_of[kj] >= 0:
            cat_mass[:, cat_of[kj]] += rows[:, j]
        if word_of[kj] >= 0:
            word_mass[:, word_of[kj]] += rows[:, j]
    pos = {int(k): i for i, k in enumerate(live)}

    cat_p = cat_mass / np.maximum(cat_mass.sum(1, keepdims=True), 1e-12)
    word_p = word_mass / np.maximum(word_mass.sum(1, keepdims=True), 1e-12)
    eps = 1e-6
    word_sm = (word_p + eps) / (word_p + eps).sum(1, keepdims=True)

    ok_c = ok_w = tot = 0
    ll = 0.0
    for t in range(len(eval_seq) - 1):
        k = int(routed[t])
        if k < 0 or k not in pos:
            continue
        i = pos[k]
        tgt_c = word_cat.get(eval_seq[t + 1])
        tgt_w = eval_seq[t + 1]
        if tgt_c is None:
            continue
        if cat_mass[i].sum() <= 0:
            continue
        ok_c += int(np.argmax(cat_p[i]) == tgt_c)
        ok_w += int(np.argmax(word_p[i]) == tgt_w)
        ll += float(np.log(word_sm[i, tgt_w]))
        tot += 1
    if tot == 0:
        return dict(nextcat_acc=0.0, nextword_acc=0.0, nextword_ll=-99.0, n=0)
    return dict(nextcat_acc=ok_c / tot, nextword_acc=ok_w / tot,
                nextword_ll=ll / tot, n=tot)


def generate(org, seed, steps=GEN_STEPS):
    """One shared generator, identical for every arm: habituated stochastic
    walk on the learned slot transition matrix. Habituation is not optional --
    without the h[k] fatigue term generation collapses onto one attractor
    (FABLE_HANDOFF, recall())."""
    live = np.where(org.used)[0]
    if len(live) < 2:
        return []
    rows = org.Pn[np.ix_(live, live)].astype(float)
    P = rows / np.maximum(rows.sum(1, keepdims=True), 1e-12)
    dead = rows.sum(1) <= 0
    rng = np.random.default_rng(90_000 + seed)
    h = np.zeros(len(live))
    cur = int(rng.integers(len(live)))
    out = []
    for _ in range(steps):
        if dead[cur]:
            cur = int(rng.integers(len(live)))
            out.append(int(live[cur]))
            continue
        w = P[cur] * np.maximum(1.0 - LAM_H * h, 1e-6)
        s = w.sum()
        nxt = int(rng.integers(len(live))) if s <= 0 else int(rng.choice(len(live), p=w / s))
        out.append(int(live[nxt]))
        h *= (1.0 - 1.0 / TAU_H)
        h[nxt] += 1.0 / TAU_H
        cur = nxt
    return out


def generation_scores(slot_seq, slot_word, slot_role, eval_seq):
    """Two arm-independent generation metrics on the WORD sequence, plus one
    slot-level diagnostic.

    gen_bigram_hit -- fraction of generated word bigrams that occur in the
        held-out corpus (phase 27 stage D's measure). Fully arm-independent.
    gen_gram_modal -- phase 8's modal-role grammaticality; dual words have no
        modal category and are skipped, so the score cannot be gamed by how
        an arm labels its slots. This is the PRIMARY generation metric.
    gen_gram_slotrole -- same grammaticality but using each slot's dominant
        TRUE role (eval only). Reported as a DIAGNOSTIC with an explicit
        caveat: a finer slot->role map is exactly what splitting produces, so
        this metric is partly circular and must not carry the verdict.
    """
    words = [slot_word[k] for k in slot_seq if k in slot_word]
    if len(words) < 10:
        return dict(gen_bigram_hit=0.0, gen_gram_modal=0.0, gen_gram_slotrole=0.0)
    corpus_bigrams = set(zip(eval_seq[:-1], eval_seq[1:]))
    hit = np.mean([(a, b) in corpus_bigrams for a, b in zip(words[:-1], words[1:])])

    ok = tot = 0
    for a, b in zip(words[:-1], words[1:]):
        ca = MODAL_CAT.get(a)
        cb = MODAL_CAT.get(b)
        if ca is None or cb is None:
            continue
        ok += int(NEXT_CAT[ca] == cb)
        tot += 1
    gram_modal = ok / max(tot, 1)

    ok2 = tot2 = 0
    for ka, kb in zip(slot_seq[:-1], slot_seq[1:]):
        ra, rb = slot_role.get(ka), slot_role.get(kb)
        if ra is None or rb is None:
            continue
        ok2 += int(NEXT_CAT[ra] == rb)
        tot2 += 1
    return dict(gen_bigram_hit=float(hit), gen_gram_modal=float(gram_modal),
                gen_gram_slotrole=ok2 / max(tot2, 1))


MODAL_CAT = {}
for _w in p10.PURE_ANIMALS:
    MODAL_CAT[p10.word_to_idx[_w]] = p10.ANIMAL
for _w in p10.PURE_ACTIONS:
    MODAL_CAT[p10.word_to_idx[_w]] = p10.ACTION
for _w in p10.OBJECTS:
    MODAL_CAT[p10.word_to_idx[_w]] = p10.OBJECT
# dual words deliberately absent: they have no fixed modal category (phase 8)


# ============================================================================
# one seed, end to end
# ============================================================================
def run_seed(seed, verbose=True):
    t0 = time.time()
    train_seq, train_roles = p10.sample_stream(SEQ_LEN, seed=99 + seed, p_dual=P_DUAL)
    eval_seq, eval_roles = p10.sample_stream(EVAL_LEN, seed=900 + seed, p_dual=P_DUAL)

    det = detect(train_seq, seed)
    detected = det['detected']
    word_cat = det['word_cat']
    if verbose:
        print(f"\n--- seed {seed}: DETECTION "
              f"(unsupervised categories k={det['n_categories']}, "
              f"silhouette {det['silhouette']:.3f}, coverage "
              f"{det['coverage']}/{N_WORDS}) ---")
        print(f"  {'word':<10}{'n':>6}{'gain':>9}{'p99 null':>10}{'margin':>9}   verdict")
        for r in sorted(det['gain_rows'], key=lambda r: -r['margin'])[:8]:
            tag = 'DUAL' if r['word'] in DUAL_IDX else ''
            print(f"  {VOCAB[r['word']]:<10}{r['n']:>6}{r['gain']:>9.3f}"
                  f"{r['null']:>10.3f}{r['margin']:>+9.3f}   "
                  f"{'DETECTED' if r['detected'] else '-':<9}{tag}")
        n_dual_det = len(detected & DUAL_IDX)
        n_ctrl_det = len(detected - DUAL_IDX)
        n_ctrl = len([r for r in det['gain_rows'] if r['word'] not in DUAL_IDX])
        print(f"  detector: {n_dual_det}/{len(DUAL_IDX)} dual words, "
              f"{n_ctrl_det}/{n_ctrl} controls false-positive at p99")

    # --- arms ---------------------------------------------------------------
    org_u, a_u = run_arm(train_seq, seed, allowed=frozenset())
    org_s, a_s = run_arm(train_seq, seed, allowed=frozenset(detected))

    occ = per_word_slot_occupancy(train_seq, a_s, org_s)
    forced = build_forced(train_seq, occ, seed)
    org_m, a_m = run_arm(train_seq, seed, forced=forced, consolidate=False)

    undetected = sorted({r['word'] for r in det['gain_rows']} - detected)
    rng_r = np.random.default_rng(30_000 + seed)
    rand_words = frozenset(rng_r.choice(undetected,
                                        size=min(len(detected), len(undetected)),
                                        replace=False).tolist()) if undetected else frozenset()
    org_r, a_r = run_arm(train_seq, seed, allowed=rand_words)

    arms = [('unsplit', org_u, a_u), ('split', org_s, a_s),
            ('matched', org_m, a_m), ('randword', org_r, a_r)]

    # --- capacity accounting (the control's whole point) ---------------------
    live = {name: int(o.used.sum()) for name, o, _ in arms}
    if verbose:
        print(f"\n--- seed {seed}: CAPACITY ---")
        for name, o, a in arms:
            spw = per_word_slot_occupancy(train_seq, a, o)
            multi = {VOCAB[w]: len(s) for w, s in spw.items() if len(s) > 1}
            print(f"  {name:<9} live slots {live[name]:>3}   "
                  f"multi-slot words: {multi if multi else '{}'}")
        match_ok = live['split'] == live['matched']
        print(f"  capacity match (split vs matched): {live['split']} vs "
              f"{live['matched']}  ->  {'EXACT' if match_ok else 'MISMATCH'}")

    # --- prediction (a): successor distinctness ------------------------------
    dist_rows = successor_distinctness(train_seq, a_s, org_s, word_cat,
                                       detected & set(occ), seed)
    dist_rows = [r for r in dist_rows if r['n_slots'] >= 2]
    if verbose:
        print(f"\n--- seed {seed}: PREDICTION (a) successor distinctness "
              f"(same-word permutation null, p99) ---")
        for r in dist_rows:
            print(f"  {VOCAB[r['word']]:<10} slots={r['n_slots']}  "
                  f"meanTV={r['stat']:.3f}  null p99={r['null']:.3f}  "
                  f"-> {'DISTINCT' if r['distinct'] else 'not distinct'}")
        if not dist_rows:
            print("  (no word carried >=2 slots -- nothing to test)")

    # --- prediction (b): downstream utility ----------------------------------
    res = {}
    for name, o, a in arms:
        sw, sc, sr = slot_maps(train_seq, a, o, word_cat, train_roles)
        routed = o.route(eval_seq, rng_seed=80_000 + seed)
        sc_pred = predict_scores(o, routed, eval_seq, sw, sc, word_cat)
        gen = generation_scores(generate(o, seed), sw, sr, eval_seq)
        res[name] = dict(live=live[name], **sc_pred, **gen)

    if verbose:
        print(f"\n--- seed {seed}: DOWNSTREAM UTILITY (held-out stream, "
              f"n={EVAL_LEN}) ---")
        hdr = (f"  {'arm':<9}{'slots':>6}{'nextcat':>9}{'nextword':>9}"
               f"{'ll/token':>10}{'gen_bigram':>11}{'gen_gram':>9}{'[slotrole]':>11}")
        print(hdr)
        for name, _, _ in arms:
            r = res[name]
            print(f"  {name:<9}{r['live']:>6}{r['nextcat_acc']:>9.3f}"
                  f"{r['nextword_acc']:>9.3f}{r['nextword_ll']:>10.3f}"
                  f"{r['gen_bigram_hit']:>11.3f}{r['gen_gram_modal']:>9.3f}"
                  f"{r['gen_gram_slotrole']:>11.3f}")
        print(f"  ({time.time() - t0:.1f}s)")

    return dict(seed=seed, detected=detected, det=det, dist_rows=dist_rows,
                res=res, live=live)


# ============================================================================
# main
# ============================================================================
def main():
    seeds = os.environ.get('PHASE41_SEEDS', '0,1,2,3,4')
    seeds = [int(s) for s in seeds.split(',') if s.strip() != '']
    print("=" * 86)
    print("PHASE 41 (T6.4) -- DETECTION DRIVES FIELD SPLITTING, SCORED AT "
          "MATCHED CAPACITY")
    print("=" * 86)
    print(f"corpus: phase 8/10 dual-role grammar, {N_WORDS} words "
          f"({len(DUAL_WORDS)} genuinely dual), train {SEQ_LEN} / held-out "
          f"{EVAL_LEN} tokens")
    print(f"mechanism: phase 10 context-primed settling at g_mem={G_MEM} "
          f"(the twice-confirmed negative is respected: perception reads the "
          f"field, it does not bend it)")
    print(f"seeds: {seeds}  (organism s, train stream 99+s, held-out 900+s)")

    runs = [run_seed(s) for s in seeds]

    print("\n" + "=" * 86)
    print("SUMMARY -- paired across seeds")
    print("=" * 86)

    n_dual_det = [len(r['detected'] & DUAL_IDX) for r in runs]
    n_fp = [len(r['detected'] - DUAL_IDX) for r in runs]
    print(f"\nDetector: dual words detected {n_dual_det} of {len(DUAL_IDX)}; "
          f"control false positives at p99 {n_fp}")

    print("\nPrediction (a) -- successor distinctness vs same-word null:")
    tot = ok = 0
    for r in runs:
        for d in r['dist_rows']:
            tot += 1
            ok += int(d['distinct'])
    print(f"  {ok}/{tot} split words show distinct successor distributions "
          f"(p99, {NULL_DRAWS} draws)"
          if tot else "  no split words formed -- prediction (a) untestable")

    print("\nCapacity (the confound T1.5 forced this phase to control):")
    for r in runs:
        print(f"  seed {r['seed']}: unsplit {r['live']['unsplit']}  "
              f"split {r['live']['split']}  matched {r['live']['matched']}  "
              f"randword {r['live']['randword']}"
              f"   {'[split==matched EXACT]' if r['live']['split'] == r['live']['matched'] else '[MISMATCH]'}")

    metrics = [('nextcat_acc', 'held-out next-category acc'),
               ('nextword_acc', 'held-out next-word acc'),
               ('nextword_ll', 'held-out next-word ll/token'),
               ('gen_bigram_hit', 'generation bigram hit'),
               ('gen_gram_modal', 'generation grammaticality (modal)')]

    print("\nPer-arm means across seeds:")
    print(f"  {'metric':<34}{'unsplit':>10}{'split':>10}{'matched':>10}{'randword':>10}")
    for key, label in metrics:
        vals = {a: np.mean([r['res'][a][key] for r in runs])
                for a in ('unsplit', 'split', 'matched', 'randword')}
        print(f"  {label:<34}{vals['unsplit']:>10.3f}{vals['split']:>10.3f}"
              f"{vals['matched']:>10.3f}{vals['randword']:>10.3f}")

    print("\nPrediction (b) -- SPLIT vs MATCHED CAPACITY "
          f"(survival: mean > +{SURVIVE_DELTA} AND no sign flip):")
    verdict_any = False
    for key, label in metrics:
        deltas = [r['res']['split'][key] - r['res']['matched'][key] for r in runs]
        m = float(np.mean(deltas))
        flip = (min(deltas) < 0) and (max(deltas) > 0)
        survives = (m > SURVIVE_DELTA) and not flip
        verdict_any = verdict_any or survives
        print(f"  {label:<34} mean {m:>+7.3f}  per-seed "
              f"[{', '.join(f'{d:+.3f}' for d in deltas)}]  "
              f"{'SURVIVES' if survives else ('sign flip' if flip else 'below threshold')}")

    print("\nFor contrast only -- SPLIT vs UNSPLIT (capacity NOT matched; "
          "T1.5 showed extra slots alone buy accuracy, so this delta is "
          "not evidence about splitting):")
    for key, label in metrics:
        deltas = [r['res']['split'][key] - r['res']['unsplit'][key] for r in runs]
        print(f"  {label:<34} mean {np.mean(deltas):>+7.3f}")

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    if verdict_any:
        print("Prediction (b) SURVIVES on at least one downstream metric: "
              "detection-driven sense splitting buys measurable downstream "
              "utility that the same slot budget spent context-blind does not.")
    else:
        print("Prediction (c) -- THE HONEST NEGATIVE -- FIRES: at matched "
              "capacity, no downstream metric clears the pre-registered "
              "survival rule. Sense splitting is a REPRESENTATIONAL NICETY "
              "on this protocol, not a functional gain. The detector is real "
              "(prediction (a)) and the split slots are real, but the "
              "capability does not follow from the detection. Reported "
              "plainly: this is a publishable negative about the polysemy "
              "line, and the split-vs-unsplit delta above must not be quoted "
              "as if it were the result -- T1.5 already priced that as "
              "capacity.")


if __name__ == '__main__':
    sys.exit(main())
