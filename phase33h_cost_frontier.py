"""
PHASE 33h (T1.9): COST-BRANCH FOLLOW-UP -- KB-per-accuracy-point parity, or
the measured floor and its reason.

Where this comes from. Phase 33g (T1.8) cut the organism's store 3.85x at
ZERO accuracy cost (K=160: 371.2KB -> 96.4KB, ACC 0.900) and repriced the
gate's cost branch without meeting it: 107.1 KB/ACC-point against the
supervised prototype bar's 35.2 (0.872 at 30.7KB), a ~3x per-point gap.
T1.9 asks for that gap closed, or for its floor measured as the owner's
decision input on re-scoping the RELEASE HOLD. Either is the deliverable.

The levers, worked in T1.9's stated priority order:
  (i)   the knee of the COMPRESSED cost curve. Phase 33d chose K on the
        UNCOMPRESSED curve; compression moved the cost-matched point from
        K=24/0.644 to K=48/0.728 in the same 30.7KB, so the
        accuracy-per-byte-optimal K has to be re-found.
  (ii)  readout richness at fixed K -- T1.6's two measured recovery targets
        (routing margin 0.061 for a soft top-m vote; preserved-but-outvoted
        label mass), eval-side, no mechanism change.
  (iii) slot-count reduction via consolidation at task boundaries: merge
        near-duplicate slots with `consolidate()`'s own rule and free the
        absorbed ones. FOLD, never re-attribute -- phase 9's negative
        (post-hoc re-sorting inherits the online mixing it meant to undo)
        stays closed, and nothing here re-runs a learning rule or moves an
        occurrence between slots.

Levers NOT worked, and why (both are settled measurements, not omissions):
representation WIDTH is exhausted -- 33g's c64 + CSR-P + float32-meta is
already lossless-to-3.85x and there is no second free halving; and low-rank
xi factorization is a MEASURED NEGATIVE (33g: seed 0 says rank-20 = 0.887
at 1.61x the bar, paired seeds 0-4 swing it -0.052..+0.033 -- best-of-grid
selection bias, exactly what T1.6 measured). Retrying it without a
reseeding-robust selection protocol would re-derive a closed dead end.

Every arm here is reseed-verified s=0-4 PAIRED, per T1.6's and T1.8's
lesson. `build_split(seed)` reseeds the split, the stream shuffles and the
organism together, and the PROTOTYPE BAR IS RECOMPUTED ON THE SAME SPLIT --
a fixed seed-0 bar against reseeded organism arms would be exactly the
unpaired comparison those two phases were burned by. seed 0 is the
committed protocol byte-for-byte, so the 33c/33d/33g anchors are checked,
not assumed.

Byte accounting. 33g's `store_bytes`/`CompressedState.nbytes` formula
verbatim (xi + P + count + age), evaluated over the LIVE slots -- lever
(iii) frees slots, and a store you would actually persist holds what is
live. On every arm that frees nothing this is bit-identical to 33g's
number, and the run asserts that rather than claiming it.

--------------------------------------------------------------------------
THE ARITHMETIC BOUND, stated BEFORE the run (it is a fact about the two
byte formulas, not a prediction, and it is what makes the honest-negative
branch of prediction (b) the likely one):

  bar         120 prototypes x N=64 REAL float32          = 256 B/memory
  organism    1 slot x N=64 COMPLEX float32 (33g's c64)   = 512 B/memory
                                     + 8 B/slot count+age + the CSR graph

A field memory is a complex vector; a prototype is a real one. At EQUAL
memory count and EQUAL accuracy the organism therefore costs >= 2.03x the
bar's bytes, before the graph. So:

  * KB/ACC-pt PARITY (1.0x, i.e. <= 30.7KB at ACC >= 0.872) requires the
    organism to clear the bar's accuracy on <= ~59 slots. K=60 measures
    0.735. Parity is arithmetically out of reach on this protocol unless
    slot count roughly halves at unchanged accuracy.
  * The <= 2x criterion T1.9 names (<= 70.4 KB/ACC-pt at ACC >= 0.872)
    needs <= 61.4KB at 0.872, i.e. <= ~115 live slots with a near-free
    graph. K=120 measures 0.854 -- just under the bar. This one is live,
    and levers (ii) and (iii) are the two shots at it.

Pre-registered predictions (T1.9's, verbatim, plus the two this phase adds
before looking at any result):
  (a) the compressed cost curve has a knee below K=160 where KB/ACC-pt
      improves but likely stays above the bar's 35.2 -- measure where.
  (b) honest negative: if no lever reaches <= 2x the bar's KB/ACC-pt at
      ACC >= 0.872, record the measured floor and the reason, as the
      decision input for an owner re-scope of the gate to the capability
      axes.
  (c) [added] lever (iii) has little to fold. Consolidation's own
      threshold is 0.8; if the slot bank is already near-orthogonal at
      that threshold, slot-count reduction is not free and every byte it
      saves is bought with accuracy. The redundancy census in section 0
      is run FIRST and decides this, the way 33g measured the P count
      distribution before pulling lever (b).
  (d) [added] survival rule for levers (ii) and (iii), fixed here so it
      cannot be chosen after the fact: an arm counts as an improvement
      only if its paired per-seed delta beats the incumbent on EVERY seed
      s=0-4 (min delta > 0), not on the mean and not on seed 0. Best-of-
      grid on one seed is the specific failure T1.6 and T1.8 both
      measured.
  (e) [added] HELD-OUT CONFIRMATION, and the reason it is not optional.
      Rule (d) is a survival filter over a grid (14 decoders x several K),
      and a filter run on the same seeds that measure the effect is still
      selection. So: any lever chosen in this phase is selected on
      s=0-4 ONLY, and the headline number is then re-measured on FRESH
      seeds s=5-9 that took no part in the choice, bar recomputed on those
      splits too. The verdict below is the HELD-OUT number. If a lever's
      gain evaporates out of sample it is 33g's low-rank story again, and
      it gets reported as such rather than banked.
--------------------------------------------------------------------------

Protocol: phase 33c's stream, scorers and prototype bar VERBATIM through
phase 33d/33e/33g's runners. K, the compression spec, the decoder and the
fold threshold are the only degrees of freedom.

COMMITTED-RUN FINDINGS (2026-08-07; numpy 2.4.6 / numba 0.66 / sklearn 1.9;
harness 45/45 both backends + equivalence + label-readout tests green before
and after; no library file touched by this phase).

Anchors: seed 0 reproduces everything it should -- the bar at ACC 0.872 /
30.7KB / 120 prototypes, 33d's ACC ladder at K = 40/80/120/160
(0.712/0.837/0.854/0.900), and 33g's compressed K=160 store at 96.4KB. The
live-slot byte accounting equals 33g's allocated accounting on every arm
that frees no slot, checked rather than asserted.

THE VERDICT: PARITY NOT MET, T1.9's <= 2x fallback NOT MET. The measured
floor is 76.1 KB/ACC-point = 2.24x the bar, at ACC 0.903 (K=112 +
`calib-b8`), on the HELD-OUT seeds s=5-9. T1.9 opened against 33g's 3.15x,
so the per-point cost fell 29% and every byte of that came from choosing K
on the compressed curve plus a free eval-side decoder -- but the gap to the
bar did not close, and the arithmetic in the header says why it cannot.

Lever (i), the knee -- prediction (a) HELD in its first clause and FAILED
in its second, in a way worth keeping. The compressed curve's knee is far
below K=160: KB/ACC-point is MINIMIZED at K=24 (23.9, i.e. 0.70x the bar)
and the organism is cheaper per point than the bar at every K <= 40. That
is not a win, and reading it as one would be the phase's easiest available
error: the metric is minimized where accuracy is worst (K=24 scores 0.583),
because accuracy per byte is steepest at the bottom of the capacity curve.
The gate's own phrase is "cost-effective AT EQUAL ACCURACY", so the
constrained number -- cheapest arm at or above the bar's accuracy -- is the
only one that answers it, and unconstrained KB/ACC-point should never be
quoted from this phase without that sentence attached. Constrained, lever
(i) alone lands at K=160 / 110.3 KB/ACC-pt / 3.25x, essentially 33g's
number: the finer grid (17 K points against 33d's 5) bought resolution, not
cost.

Lever (ii), readout richness -- the phase's real finding, and it BOUNDS
T1.6's null rather than overturning it. T1.6 measured its null at K=40 and
concluded "memory content, not readout geometry, is the gate-gap limit."
That conclusion is correct AT K=40 and this run reproduces its shape (at
K=24 nothing survives; best mean delta +0.009). But the null is
capacity-dependent: at K=120 five decoders clear the survival rule on all
five seeds -- dist-m2/m3/m5 and calib-b8/b32 -- with calib-b8 at +0.053
[+0.012, +0.081]. The mechanism is the one T1.6 named and could not
exercise: per-slot label DISTRIBUTIONS instead of majority collapse, which
only has something to recover once there are enough slots for label mass to
be split rather than concentrated. Composed with lever (i), calib-b8 moves
the whole curve down at zero byte cost and the cheapest at-bar arm goes
K=160/3.25x -> K=96/1.94x on the selection seeds.

...and then rule (e) took most of it back, which is exactly why rule (e)
exists. On fresh seeds s=5-9 (bar recomputed on those splits), K=96's
decoder gain SIGN-FLIPS -- +0.023 mean but min -0.004 -- so its 1.96x is
recorded and NOT banked; a 14-decoder x 3-K grid filtered on the same seeds
that measure it is still selection, and 33g's low-rank arm is the standing
example of what that produces. The point that survives out of sample is
K=112: +0.026 [+0.009, +0.050], positive on every held-out seed, ACC 0.903
against a held-out bar of 0.865, at 68.7KB = 76.1 KB/ACC-pt = 2.24x. That
is the number in the verdict.

Lever (iii), consolidation folding -- prediction (c) HELD, and the census
called it before a single fold ran. At consolidate()'s own merge_thresh of
0.8 the slot bank is already near-orthogonal: 8 duplicate pairs out of
12 720 at K=160 (13 slots of 160 have any partner at all; median pairwise
overlap 0.169, max 0.883). There is essentially nothing to fold for free,
so every byte this lever saves is bought with accuracy, and the sweep
prices that purchase as a bad one at the end that matters: folding hard
enough to matter (th <= 0.60, 160 -> 132 live slots in store mode) costs
-0.147 ACC and drops the arm far below the bar, while folding gently enough
to keep the bar (th = 0.85) frees 0.2 slots on average and moves 3.22x to
3.25x -- noise. No fold arm anywhere in the sweep is both at/above the bar
and cheaper per point than not folding. Store-mode and eval-only agree on
the shape and differ as expected (compounded loss is worse: at th=0.70,
store -0.057 vs eval-only -0.045).

WHY THE FLOOR IS WHERE IT IS -- the reason T1.9 asked for, and it is
arithmetic, not a tuning failure. A prototype is a REAL float32 N-vector
(256 B); a field memory is a COMPLEX one (512 B) plus 8 B of count/age.
That is 2.03x per stored memory before the transition graph costs a byte,
and it is not a storage-engineering target: the complex state IS the
mechanism -- phase carries the binding and sequence structure that the
whole architecture rests on, and phase 33g already showed the amplitude
side is at its lossless floor (c64 + CSR, 3.85x, zero accuracy cost).
The second factor multiplies it: the organism needs 112 slots to clear a
bar that 115 prototypes clear (held-out mean), so slot efficiency is
roughly at parity and buys nothing back. The floor decomposes exactly,
every term measured:

    68.7KB / 29.4KB = 2.34x bytes    x  0.865/0.903 = 0.958 accuracy
                                     -> 2.24x KB/ACC-pt
    of which:  2.03 x (112/115) = 1.98x   complex-vs-real, at parity count
               + 10.5KB / 29.4KB = 0.36x  the CSR transition graph

with no slack left in any term this phase can move. Concretely: parity
would require clearing the bar on <= 57 slots (K=56 measures 0.800), and
even the <= 2x fallback needs <= 114 slots with a FREE graph -- the
held-out 2x budget is 58.8KB and K=112 spends 68.7KB, overshooting by
9.9KB against a graph that costs 10.5KB. The fallback was missed by
almost exactly the size of the transition matrix, which is the one part
of the store this benchmark's readout never even reads (33g's integrity
note) but phase 30's reasoning consumers do.

ONE SCOPE VARIATION, reported because it is decision-relevant and NOT
banked as a lever. This benchmark's readout scores label evidence over xi
overlaps and never queries the transition graph, so a store that dropped
the graph entirely would cost 58.2KB at ACC 0.903 = 64.5 KB/ACC-pt =
1.90x -- it would MEET the fallback. That number is honest only for a
product that never reasons. The graph IS the logic layer (phase 30's
kstep/rollout/plan, phases 35/36's hierarchical routing), and it is what
every capability the gate would be re-scoped TOWARD is built on; quoting
1.90x without this paragraph attached would be precisely the measurement
artifact 33g refused to bank when P pruning came out "free". The verdict
above therefore stands at 2.24x, and 1.90x is on record as the floor for
a classifier-only episodic store.

WHAT THIS BUYS THE OWNER'S DECISION. The cost branch is not closable by
storage engineering: 33g took the lossless width, 33h took the free
readout and the right K, and what remains is the complex-vs-real factor,
which is the premise rather than an implementation. The gate's cost branch
as literally written ("cost-effective at equal accuracy" vs a supervised
prototype bar) therefore has a measured floor of ~2.2x, and re-scoping it
to the capability axes -- unsupervised representation + structure learning
+ bitwise persistence in one single-pass online mechanism, none of which
the prototype bar does at all -- is a decision the data now supports rather
than merely permits.

One comparison that moved, recorded with its caveat: 33c's replay arm tops
the committed ladder at 0.913 and costs ~89.6KB = 98.1 KB/ACC-pt (2.89x the
bar). The K=112 arm here is 0.903 at 76.1 KB/ACC-pt -- the organism is now
CHEAPER PER ACCURACY POINT THAN REPLAY at nearly its accuracy, having been
3.4x dearer at 33d. Caveat, load-bearing: replay's numbers are 33c's
committed torch run at seed 0 and were NOT reseeded or re-measured here
(torch is not installed on this host), so this is a comparison against a
single-seed number and the organism's own single-seed spread on this
protocol is wide. It is a direction, not a claim.
"""

import copy
import time

import numpy as np

import phase33e_readout_geometry as p33e
from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_compress import CompressionSpec, compress, store_bytes
from organism_numba import NumbaOrganism
from phase33c_gate_retest import EVICT, N, NORM, TASKS, X, task_acc_matrix_to_metrics

build_split = p33e.build_split          # seed 0 == the committed 33c protocol

SEEDS = (0, 1, 2, 3, 4)          # selection + measurement (seed 0 = committed)
SEEDS_HELD = (5, 6, 7, 8, 9)     # held out: never used to choose anything

# 33g's headline compression arm: c64 xi + CSR P (floor 0, lossless) +
# float32 count/age. The whole 3.85x, at measured zero accuracy cost.
SPEC = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0, meta_dtype=np.float32)

# lever (i): 33d's five K points plus the intermediate ones its coarse grid
# skipped -- the knee cannot be seen on a grid whose gaps are 40 wide
K_FRONTIER = [24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 120, 128, 144, 160,
              176, 192, 224]

# lever (iii): consolidate()'s default merge_thresh is 0.8; the sweep goes
# below it to trace what merging genuinely distinct slots costs
FOLD_THRESH = [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

# committed anchors that must reproduce at seed 0
ANCHOR_33D_ACC = {40: 0.712, 60: 0.735, 80: 0.837, 120: 0.854, 160: 0.900}
ANCHOR_33G_K160_BYTES = 96.4e3       # c64 + CSR + meta32 at K=160
ANCHOR_33G_K160_ACC = 0.900
ANCHOR_PROTO = (0.872, 30720)        # 33c's bar: ACC, bytes (120 x N x 4)
REPLAY_BYTES = 89.6e3                # 33c committed: 38.4KB params + 51.2KB buffer

BAR_RATIO_TARGET = 2.0               # T1.9's "<= 2x the bar's KB/ACC-pt"
CLAIM_THRESH = p33e.CLAIM_THRESH     # T1.6's 0.02 ACC claim threshold


# ---------------------------------------------------------------------------
# byte accounting: 33g's formula over the LIVE slots
# ---------------------------------------------------------------------------

class StoreView:
    """The four accounted fields restricted to a slot subset -- what a store
    you would actually persist holds. `compress`/`store_bytes` read exactly
    (K, N, xi, P, count, age, used), so a view is all they need.

    On arms that free no slots this is the whole store and the byte number
    is bit-identical to 33g's; `check_view_identity` asserts that."""

    def __init__(self, org, keep=None):
        keep = np.where(org.used)[0] if keep is None else np.asarray(keep)
        self.K, self.N = int(len(keep)), int(org.N)
        self.xi = np.ascontiguousarray(org.xi[keep])
        self.P = np.ascontiguousarray(np.asarray(org.P)[np.ix_(keep, keep)])
        self.count = np.asarray(org.count)[keep]
        self.age = np.asarray(org.age)[keep]
        self.used = np.asarray(org.used, bool)[keep]


def live_bytes(org, spec=SPEC):
    """Compressed store bytes over the live slots (33g's accounting)."""
    return int(compress(StoreView(org), spec=spec).nbytes)


# ---------------------------------------------------------------------------
# lever (iii): duplicate folding -- consolidate()'s rule, made destructive
# ---------------------------------------------------------------------------

def redundancy_census(org, thresholds):
    """Pairwise |<xi_i, xi_j>|/N over the live slots: how much duplication
    is there to fold BEFORE any lever is pulled (prediction (c))."""
    idx = np.where(org.used)[0]
    xi = np.ascontiguousarray(org.xi, dtype=complex)[idx]
    O = np.abs(xi @ xi.conj().T) / org.N
    np.fill_diagonal(O, 0.0)
    off = O[np.triu_indices(len(idx), 1)]
    rows = []
    for th in thresholds:
        rows.append((th, int((off > th).sum()), int((O > th).any(1).sum())))
    return len(idx), off, rows


def fold_duplicates(org, readout, thresh):
    """Merge near-duplicate slots and FREE the absorbed ones.

    The pairing rule is `Organism.consolidate`'s own, verbatim: a greedy
    first-duplicate scan over the live slots' pairwise overlap matrix, at
    the same `merge_thresh` semantics. The difference is only what happens
    after the pair is found -- consolidate builds a compact `mem`/`Pn` view
    and leaves the store alone, while this frees the absorbed slot so the
    STORE shrinks and the freed capacity is recruitable again.

    Every mutation is a fold along an existing identity, mirroring the
    EventBoundary fusion contract that `organism.py` already emits:
      * `graph.merge(keep, drop)` -- drop's relations fold into keep, then
        drop is retired (the graph's own fusion op).
      * `count[keep] += count[drop]` -- one identity, one lifetime count.
      * `readout.remap(drop, keep)` -- label evidence follows the identity
        (LabelEvidenceReadout's documented lifecycle call).
      * `xi[keep]` is left UNTOUCHED, exactly as consolidate keeps the
        first slot's vector rather than re-averaging.

    Nothing re-attributes an occurrence to a different slot and no learning
    rule runs: this re-encodes a finished state under a coarser identity
    map. Phase 9's negative is about post-hoc RE-ATTRIBUTION and is neither
    invoked nor re-opened.

    Returns the number of slots freed.
    """
    idx = np.where(org.used)[0]
    if len(idx) < 2:
        return 0
    xi = np.ascontiguousarray(org.xi, dtype=complex)[idx]
    O = np.abs(xi @ xi.conj().T) / org.N
    kept, kept_pos, pairs = [], [], []
    for pk, k in enumerate(idx):
        dup = next((j for pj, j in zip(kept_pos, kept) if O[pk, pj] > thresh),
                   None)
        if dup is None:
            kept.append(int(k)); kept_pos.append(pk)
        else:
            pairs.append((int(k), int(dup)))          # (drop, keep)
    for drop, keep in pairs:
        org.graph.merge(keep, drop)                   # folds, then retires drop
        org.count[keep] += org.count[drop]
        org.used[drop] = False
        org.count[drop] = 0.0
        org.age[drop] = 0.0
        org.xi[drop] = 0.0
        readout.remap(drop, keep)
    return len(pairs)


# ---------------------------------------------------------------------------
# the runner -- 33g's `run_arm` with the fold hook and per-decoder scoring
# ---------------------------------------------------------------------------

def run_arm(K, seed=0, evict=EVICT, spec=SPEC, fold=None, fold_mode="store",
            decoders=(("argmax", dict(decoder="argmax")),), hold=8, epochs=3):
    """Phase 33g's `run_arm` (itself 33d's `run_organism_K`) with two hooks.

    fold        None, or a merge threshold for lever (iii)
    fold_mode   'store'    fold at every task boundary and carry it forward
                           (the deployment path; loss compounds, and the
                           freed slots are recruitable by later tasks)
                'evalonly' fold a COPY just before the final scoring
                           (isolates the storage gain from the compounded
                           one, exactly as 33g split store vs eval-only)
    decoders    scored simultaneously off one trained organism (33e's
                technique) -- every decoder sees identical organism state,
                so decoder deltas are paired by construction
    """
    Xtr, ytr, Xte, yte = build_split(seed)
    r = np.random.default_rng(seed)
    r.permutation(len(X))                    # burn the split draw (33b's technique)
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=seed)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    A = {name: np.zeros((len(TASKS), len(TASKS))) for name, _ in decoders}
    freed = 0
    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in r.permutation(idx):
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict)
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        readout.invalidate(fresh)            # readout follows slot identity
        readout.observe(org, Xtr[idx], ytr[idx])
        if spec is not None:                 # quantize the store and carry it
            compress(org, spec=spec).apply(org)
        scored, scored_readout = org, readout
        if fold is not None:
            if fold_mode == "store":
                freed += fold_duplicates(org, readout, fold)
            else:                            # eval-only: fold a copy, keep learning wide
                scored = copy.deepcopy(org)
                scored_readout = copy.deepcopy(readout)
                freed = fold_duplicates(scored, scored_readout, fold)
        for name, kw in decoders:
            A[name][ti] = p33e.eval_tasks(
                lambda Xe, s=scored, ro=scored_readout, kw=kw: ro.predict(s, Xe, **kw),
                Xte, yte)
    out = {}
    for name, _ in decoders:
        acc, forg = task_acc_matrix_to_metrics(A[name])
        out[name] = dict(A=A[name], acc=acc, forg=forg)
    return dict(per_decoder=out, org=org, readout=readout, freed=freed,
                scored=scored if fold is not None else org,
                live=int(org.used.sum()),
                bytes=live_bytes(scored if fold is not None else org, spec),
                alloc_bytes=int(compress(org, spec=spec).nbytes)
                if spec is not None else store_bytes(org))


def run_prototypes_seed(seed, thresh=0.55):
    """Phase 33c's supervised prototype bar, VERBATIM, on `build_split(seed)`.

    33c's own `run_prototypes` is pinned to the module-level seed-0 split.
    Reseeded organism arms have to be scored against a bar drawn from the
    SAME split or the comparison is unpaired -- the precise error T1.6 and
    T1.8 measured on the organism side. seed 0 reproduces 33c's 0.872/120.
    """
    Xtr, ytr, Xte, yte = build_split(seed)
    protos, labels = [], []
    A = np.zeros((len(TASKS), len(TASKS)))
    for ti, task in enumerate(TASKS):
        idx = np.where(np.isin(ytr, task))[0]
        for i in idx:
            v = Xtr[i] / (np.linalg.norm(Xtr[i]) + 1e-9)
            if protos:
                sims = np.array([v @ p for p in protos])
                j = int(np.argmax(sims))
                if sims[j] > thresh:
                    protos[j] = protos[j] + 0.1 * (v - protos[j])
                    protos[j] /= np.linalg.norm(protos[j]) + 1e-9
                    continue
            protos.append(v); labels.append(ytr[i])

        def pred(Xe, P=np.array(protos), L=np.array(labels)):
            Xn = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9)
            return L[np.argmax(Xn @ P.T, axis=1)]
        A[ti] = p33e.eval_tasks(pred, Xte, yte)
    acc, forg = task_acc_matrix_to_metrics(A)
    return dict(acc=acc, forg=forg, bytes=len(protos) * N * 4,
                n=len(protos), kbpp=(len(protos) * N * 4 / 1e3) / acc)


def kbpp(bytes_, acc):
    """KB per accuracy point -- the gate's second-branch metric."""
    return (bytes_ / 1e3) / acc if acc > 0 else float("inf")


def paired(values):
    """(mean, min, max) of a paired per-seed series."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.min()), float(v.max())


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    print("PHASE 33h (T1.9): the cost frontier -- KB/ACC-point parity or its "
          "measured floor\n")
    print(f"  seeds {list(SEEDS)} paired; compression = {SPEC.label()} "
          f"(33g's 3.85x arm, measured zero-cost)")
    print(f"  bar recomputed per seed on the SAME split (paired comparison)\n")

    # -- the bar, per seed ------------------------------------------------
    bars = {s: run_prototypes_seed(s) for s in SEEDS}
    b0 = bars[0]
    ok = abs(b0['acc'] - ANCHOR_PROTO[0]) < 5e-4 and b0['bytes'] == ANCHOR_PROTO[1]
    print("the bar (supervised growing prototypes), per seed:")
    for s in SEEDS:
        b = bars[s]
        print(f"  seed {s}: ACC {b['acc']:.3f}  FORG {b['forg']:.3f}  "
              f"{b['bytes'] / 1e3:5.1f}KB ({b['n']} protos)  "
              f"{b['kbpp']:5.1f} KB/ACC-pt")
    bar_kbpp = paired([bars[s]['kbpp'] for s in SEEDS])
    bar_acc = paired([bars[s]['acc'] for s in SEEDS])
    print(f"  seed-0 anchor (33c: ACC 0.872 / 30.7KB / 120 protos): "
          f"{'REPRODUCED' if ok else 'MISMATCH'}")
    print(f"  paired bar: ACC {bar_acc[0]:.3f} [{bar_acc[1]:.3f}, {bar_acc[2]:.3f}]"
          f"   {bar_kbpp[0]:.1f} KB/ACC-pt [{bar_kbpp[1]:.1f}, {bar_kbpp[2]:.1f}]")
    print(f"  targets: parity = {bar_kbpp[0]:.1f} KB/ACC-pt at ACC >= the bar; "
          f"T1.9's fallback = {BAR_RATIO_TARGET:.0f}x = "
          f"{BAR_RATIO_TARGET * bar_kbpp[0]:.1f}\n")

    # -- the arithmetic bound, restated against the measured bar ----------
    per_slot = 8 * N + 8                     # c64 xi + float32 count + float32 age
    per_proto = 4 * N
    budget2x = BAR_RATIO_TARGET * bar_kbpp[0] * bar_acc[0] * 1e3
    print("the arithmetic bound (stated in the docstring, now against the "
          "measured bar):")
    print(f"  a prototype is a REAL N={N} float32 vector      "
          f"{per_proto} B/memory")
    print(f"  a slot is a COMPLEX N={N} float32 vector + meta "
          f"{per_slot} B/memory  ({per_slot / per_proto:.2f}x)")
    print(f"  parity budget at ACC {bar_acc[0]:.3f}: "
          f"{bar_kbpp[0] * bar_acc[0]:.1f}KB = "
          f"{int(bar_kbpp[0] * bar_acc[0] * 1e3 / per_slot)} slots, graph free")
    print(f"  {BAR_RATIO_TARGET:.0f}x budget:                    "
          f"{budget2x / 1e3:.1f}KB = {int(budget2x / per_slot)} slots, graph free\n")

    # =====================================================================
    # SECTION 0 -- redundancy census (prediction (c): decides lever (iii))
    # =====================================================================
    print("=" * 74)
    print("SECTION 0 -- redundancy census: is there anything to fold? "
          "(prediction (c))")
    print("=" * 74)
    print("pairwise |<xi_i, xi_j>|/N over live slots; consolidate()'s own "
          "merge_thresh is 0.8\n")
    for K in (120, 160):
        base = run_arm(K, seed=0)
        n, off, rows = redundancy_census(base['org'], [0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
        print(f"  K={K:>3} ({n} live): max {off.max():.3f}  p99 "
              f"{np.quantile(off, 0.99):.3f}  p95 {np.quantile(off, 0.95):.3f}  "
              f"median {np.median(off):.3f}")
        for th, npair, nslot in rows:
            print(f"        pairs > {th:.2f}: {npair:>4}   slots with a "
                  f"partner: {nslot:>4} / {n}")
    print()

    # =====================================================================
    # SECTION A -- lever (i): the knee of the COMPRESSED cost curve
    # =====================================================================
    print("=" * 74)
    print("SECTION A -- lever (i): the knee of the compressed cost curve "
          "(prediction (a))")
    print("=" * 74)
    print("33d chose K on the UNCOMPRESSED curve; this re-finds it under "
          "33g's compression.\n")
    print(f"  {'K':>4}  {'ACC (mean [min,max])':<28} {'FORG':>6}  "
          f"{'live':>4} {'KB':>7}  {'KB/ACC-pt':>10}  {'vs bar':>7}  anchor")
    frontier = {}
    for K in K_FRONTIER:
        runs = [run_arm(K, seed=s) for s in SEEDS]
        accs = [r['per_decoder']['argmax']['acc'] for r in runs]
        forgs = [r['per_decoder']['argmax']['forg'] for r in runs]
        byts = [r['bytes'] for r in runs]
        kbs = [kbpp(r['bytes'], a) for r, a in zip(runs, accs)]
        am, alo, ahi = paired(accs)
        km, klo, khi = paired(kbs)
        # the live-slot view must equal 33g's allocated accounting when the
        # arm frees nothing -- asserted, not assumed
        view_ok = all(r['bytes'] == r['alloc_bytes'] for r in runs)
        frontier[K] = dict(runs=runs, accs=accs, kbs=kbs, bytes=byts,
                           acc=am, kbpp=km, forg=float(np.mean(forgs)),
                           ratio=km / bar_kbpp[0], view_ok=view_ok)
        anchor = ""
        if K in ANCHOR_33D_ACC:
            hit = abs(accs[0] - ANCHOR_33D_ACC[K]) < 5e-4
            anchor = (f"33d {ANCHOR_33D_ACC[K]:.3f} "
                      f"{'OK' if hit else 'MISMATCH'}")
        if K == 160:
            hit_b = abs(byts[0] - ANCHOR_33G_K160_BYTES) / ANCHOR_33G_K160_BYTES < 0.01
            anchor += f" | 33g {ANCHOR_33G_K160_BYTES / 1e3:.1f}KB " \
                      f"{'OK' if hit_b else 'MISMATCH'}"
        star = " *" if am >= bar_acc[0] else "  "
        print(f"  {K:>4}  {am:.3f} [{alo:.3f}, {ahi:.3f}]{star}       "
              f"{np.mean(forgs):.3f}  {int(np.mean([r['live'] for r in runs])):>4} "
              f"{np.mean(byts) / 1e3:7.1f}  {km:10.1f}  "
              f"{km / bar_kbpp[0]:6.2f}x  {anchor}")
    print("  (* = mean ACC at or above the paired bar)")
    bad_view = [K for K in K_FRONTIER if not frontier[K]['view_ok']]
    print(f"  live-slot accounting == 33g's allocated accounting on every "
          f"non-folding arm: {'YES' if not bad_view else f'NO at K={bad_view}'}")

    knee = min(K_FRONTIER, key=lambda K: frontier[K]['kbpp'])
    print(f"\n  KNEE (min KB/ACC-pt, any accuracy): K={knee} at "
          f"{frontier[knee]['kbpp']:.1f} KB/ACC-pt "
          f"({frontier[knee]['ratio']:.2f}x the bar), ACC "
          f"{frontier[knee]['acc']:.3f}")
    above = [K for K in K_FRONTIER if frontier[K]['acc'] >= bar_acc[0]]
    if above:
        best = min(above, key=lambda K: frontier[K]['kbpp'])
        print(f"  CHEAPEST AT/ABOVE THE BAR: K={best} at "
              f"{frontier[best]['kbpp']:.1f} KB/ACC-pt "
              f"({frontier[best]['ratio']:.2f}x), ACC {frontier[best]['acc']:.3f}, "
              f"{np.mean(frontier[best]['bytes']) / 1e3:.1f}KB")
        lever_i = (best, frontier[best])
    else:
        print("  no swept K reaches the paired bar's accuracy")
        lever_i = None
    print()

    # =====================================================================
    # SECTION B -- lever (ii): readout richness at fixed K
    # =====================================================================
    print("=" * 74)
    print("SECTION B -- lever (ii): readout richness at fixed K "
          "(T1.6's recovery targets)")
    print("=" * 74)
    print("Eval-side only; bytes are FIXED, so any ACC gain is a pure "
          "KB/ACC-pt gain.")
    print("T1.6's verdict was a robust NULL at K=40; the open question is "
          "whether it")
    print("still holds at the capacities that matter to the cost branch. "
          "Survival rule (d):")
    print("paired min delta > 0 on ALL of s=0-4 AND mean delta > "
          f"{CLAIM_THRESH:.2f}.\n")
    K_READOUT = sorted(set([knee, 120, 160] + ([lever_i[0]] if lever_i else [])))
    decoder_rows = {}
    for K in K_READOUT:
        runs = [run_arm(K, seed=s, decoders=p33e.DECODERS) for s in SEEDS]
        base = np.array([r['per_decoder']['argmax']['acc'] for r in runs])
        byts = float(np.mean([r['bytes'] for r in runs]))
        print(f"  K={K}  (argmax baseline ACC {base.mean():.3f}, "
              f"{byts / 1e3:.1f}KB, {kbpp(byts, base.mean()):.1f} KB/ACC-pt)")
        best_row = None
        for name, _ in p33e.DECODERS:
            if name == "argmax":
                continue
            a = np.array([r['per_decoder'][name]['acc'] for r in runs])
            d = a - base
            survives = bool(d.min() > 0 and d.mean() > CLAIM_THRESH)
            decoder_rows[(K, name)] = dict(acc=a, delta=d, survives=survives)
            flag = "  <== SURVIVES rule (d)" if survives else ""
            if best_row is None or d.mean() > best_row[1]:
                best_row = (name, d.mean(), d.min(), d.max(), a.mean())
            print(f"     {name:<10} ACC {a.mean():.3f}   delta "
                  f"{d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}]"
                  f"   KB/ACC-pt {kbpp(byts, a.mean()):6.1f}{flag}")
        nm, dm, dlo, dhi, am = best_row
        print(f"     -> best mean delta: {nm} {dm:+.3f} "
              f"[{dlo:+.3f}, {dhi:+.3f}]; "
              f"{'CLAIM' if dlo > 0 and dm > CLAIM_THRESH else 'no robust claim'}\n")
    survivors = [k for k, v in decoder_rows.items() if v['survives']]
    print(f"  decoders surviving rule (d) anywhere: "
          f"{survivors if survivors else 'NONE -- T1.6 null holds at every K tested'}\n")

    # -- selection, on s=0-4 ONLY (rule (e)) ------------------------------
    selected = None
    if survivors:
        # declared before the run: among rule-(d) survivors, take the largest
        # mean paired delta. One name, chosen once, on the selection seeds.
        (K_sel, name_sel) = max(survivors,
                                key=lambda k: decoder_rows[k]['delta'].mean())
        selected = dict(name=name_sel,
                        kw=dict(p33e.DECODERS)[name_sel],
                        K_at_selection=K_sel,
                        delta=decoder_rows[(K_sel, name_sel)]['delta'])
        print(f"  SELECTED (on s=0-4 only): decoder '{name_sel}' "
              f"(chosen at K={K_sel}, mean delta "
              f"{selected['delta'].mean():+.3f} "
              f"[{selected['delta'].min():+.3f}, "
              f"{selected['delta'].max():+.3f}])")
        print(f"  T1.6 measured its null at K=40. This is the same decoder "
              f"family at higher\n  capacity -- so the null is not overturned, "
              f"it is bounded: see section B2.\n")

    # =====================================================================
    # SECTION B2 -- levers (i) x (ii) composed: the decoder-equipped curve
    # =====================================================================
    curve2, at_bar2, K_best2 = {}, [], None
    if selected is not None:
        print("=" * 74)
        print(f"SECTION B2 -- levers (i) x (ii) composed: the whole cost "
              f"curve under '{selected['name']}'")
        print("=" * 74)
        print("Lever (ii) costs ZERO bytes, so it moves every point on lever "
              "(i)'s curve straight")
        print("down the KB/ACC-pt axis. The cheapest at-bar arm has to be "
              "re-found on THIS curve.\n")
        dec = [("argmax", dict(decoder="argmax")),
               (selected['name'], selected['kw'])]
        print(f"  {'K':>4}  {'argmax':>7}  {selected['name']:>10}  "
              f"{'delta (min,max)':<24} {'KB':>7}  {'KB/ACC-pt':>10}  {'vs bar':>7}")
        for K in K_FRONTIER:
            runs = [run_arm(K, seed=s, decoders=dec) for s in SEEDS]
            a0 = np.array([r['per_decoder']['argmax']['acc'] for r in runs])
            a1 = np.array([r['per_decoder'][selected['name']]['acc']
                           for r in runs])
            byts = float(np.mean([r['bytes'] for r in runs]))
            d = a1 - a0
            k1 = kbpp(byts, a1.mean())
            curve2[K] = dict(acc=float(a1.mean()), accs=a1, bytes=byts,
                             kbpp=k1, ratio=k1 / bar_kbpp[0],
                             delta=d, robust=bool(d.min() > 0))
            star = " *" if a1.mean() >= bar_acc[0] else "  "
            print(f"  {K:>4}  {a0.mean():7.3f}  {a1.mean():10.3f}{star} "
                  f"{d.mean():+.3f} [{d.min():+.3f}, {d.max():+.3f}]      "
                  f"{byts / 1e3:7.1f}  {k1:10.1f}  {k1 / bar_kbpp[0]:6.2f}x")
        at_bar2 = [K for K in K_FRONTIER if curve2[K]['acc'] >= bar_acc[0]]
        if at_bar2:
            K_best2 = min(at_bar2, key=lambda K: curve2[K]['kbpp'])
            print(f"\n  CHEAPEST AT/ABOVE THE BAR on the composed curve: "
                  f"K={K_best2} at {curve2[K_best2]['kbpp']:.1f} KB/ACC-pt "
                  f"({curve2[K_best2]['ratio']:.2f}x), ACC "
                  f"{curve2[K_best2]['acc']:.3f}, "
                  f"{curve2[K_best2]['bytes'] / 1e3:.1f}KB")
        else:
            K_best2 = None
            print("\n  no K on the composed curve reaches the paired bar")
        print()

    # =====================================================================
    # SECTION B3 -- HELD-OUT confirmation on fresh seeds (rule (e))
    # =====================================================================
    held = {}
    if selected is not None and at_bar2:
        print("=" * 74)
        print("SECTION B3 -- held-out confirmation on fresh seeds "
              f"{list(SEEDS_HELD)} (rule (e))")
        print("=" * 74)
        print("Nothing below took any part in choosing the decoder or K. "
              "The bar is recomputed")
        print("on these splits too, so the comparison stays paired. THIS is "
              "the verdict's number.\n")
        bars_h = {s: run_prototypes_seed(s) for s in SEEDS_HELD}
        bar_acc_h = paired([bars_h[s]['acc'] for s in SEEDS_HELD])
        bar_kbpp_h = paired([bars_h[s]['kbpp'] for s in SEEDS_HELD])
        print(f"  held-out bar: ACC {bar_acc_h[0]:.3f} "
              f"[{bar_acc_h[1]:.3f}, {bar_acc_h[2]:.3f}]   "
              f"{bar_kbpp_h[0]:.1f} KB/ACC-pt "
              f"[{bar_kbpp_h[1]:.1f}, {bar_kbpp_h[2]:.1f}]\n")
        dec = [("argmax", dict(decoder="argmax")),
               (selected['name'], selected['kw'])]
        # the chosen point, plus its two neighbours on the composed curve so
        # the report is a small frontier rather than a single lucky K
        order = sorted(at_bar2, key=lambda K: curve2[K]['kbpp'])[:3]
        for K in order:
            runs = [run_arm(K, seed=s, decoders=dec) for s in SEEDS_HELD]
            a0 = np.array([r['per_decoder']['argmax']['acc'] for r in runs])
            a1 = np.array([r['per_decoder'][selected['name']]['acc']
                           for r in runs])
            byts = float(np.mean([r['bytes'] for r in runs]))
            d = a1 - a0
            k1 = kbpp(byts, a1.mean())
            held[K] = dict(acc=float(a1.mean()), bytes=byts, kbpp=k1,
                           ratio=k1 / bar_kbpp_h[0], delta=d,
                           at_bar=bool(a1.mean() >= bar_acc_h[0]),
                           robust=bool(d.min() > 0))
            print(f"  K={K:>3}  argmax {a0.mean():.3f} -> "
                  f"{selected['name']} {a1.mean():.3f}   delta {d.mean():+.3f} "
                  f"[{d.min():+.3f}, {d.max():+.3f}]"
                  f"   {'holds on every held-out seed' if d.min() > 0 else 'SIGN-FLIPS out of sample'}")
            print(f"        {byts / 1e3:.1f}KB  {k1:.1f} KB/ACC-pt = "
                  f"{k1 / bar_kbpp_h[0]:.2f}x the held-out bar   "
                  f"{'AT/ABOVE bar' if a1.mean() >= bar_acc_h[0] else 'below bar'}")
        print()
    else:
        bar_acc_h = bar_kbpp_h = None

    # =====================================================================
    # SECTION C -- lever (iii): consolidation folding at task boundaries
    # =====================================================================
    print("=" * 74)
    print("SECTION C -- lever (iii): slot-count reduction by duplicate "
          "folding")
    print("=" * 74)
    print("consolidate()'s greedy first-duplicate rule, made destructive so "
          "the STORE shrinks.")
    print("Fold only: graph.merge + count fold + readout.remap along an "
          "existing identity;")
    print("xi is never re-averaged and no occurrence moves slots (phase 9's "
          "negative stays closed).\n")
    K_FOLD = sorted(set([160, 192] + ([lever_i[0]] if lever_i else [])))
    fold_rows = {}
    for K in K_FOLD:
        ref = frontier[K] if K in frontier else None
        if ref is None:
            runs0 = [run_arm(K, seed=s) for s in SEEDS]
            ref = dict(acc=float(np.mean([r['per_decoder']['argmax']['acc']
                                          for r in runs0])),
                       kbpp=float(np.mean([kbpp(r['bytes'],
                                                r['per_decoder']['argmax']['acc'])
                                           for r in runs0])),
                       bytes=[r['bytes'] for r in runs0],
                       accs=[r['per_decoder']['argmax']['acc'] for r in runs0])
        print(f"  K={K}  unfolded: ACC {ref['acc']:.3f}  "
              f"{np.mean(ref['bytes']) / 1e3:.1f}KB  "
              f"{ref['kbpp']:.1f} KB/ACC-pt ({ref['kbpp'] / bar_kbpp[0]:.2f}x)")
        for mode in ("store", "evalonly"):
            for th in FOLD_THRESH:
                runs = [run_arm(K, seed=s, fold=th, fold_mode=mode)
                        for s in SEEDS]
                accs = [r['per_decoder']['argmax']['acc'] for r in runs]
                byts = [r['bytes'] for r in runs]
                lives = [r['live'] if mode == "store"
                         else int(r['scored'].used.sum()) for r in runs]
                kbs = [kbpp(b, a) for b, a in zip(byts, accs)]
                dacc = np.array(accs) - np.array(ref['accs'])
                am, alo, ahi = paired(accs)
                km = float(np.mean(kbs))
                better = km < ref['kbpp'] and dacc.min() > -CLAIM_THRESH
                fold_rows[(K, mode, th)] = dict(
                    acc=am, kbpp=km, bytes=float(np.mean(byts)),
                    live=float(np.mean(lives)), dacc=dacc,
                    ratio=km / bar_kbpp[0])
                mark = ""
                if am >= bar_acc[0]:
                    mark += " *bar"
                if better:
                    mark += " <cheaper/pt"
                print(f"     {mode:<8} th={th:.2f}  live {np.mean(lives):5.1f}  "
                      f"ACC {am:.3f} [{alo:.3f}, {ahi:.3f}]  dACC "
                      f"{dacc.mean():+.3f} [{dacc.min():+.3f}, {dacc.max():+.3f}]  "
                      f"{np.mean(byts) / 1e3:6.1f}KB  {km:6.1f} KB/ACC-pt  "
                      f"{km / bar_kbpp[0]:5.2f}x{mark}")
        print()

    # =====================================================================
    # SECTION D -- the frontier and the verdict
    # =====================================================================
    print("=" * 74)
    print("SECTION D -- the measured cost frontier and the verdict")
    print("=" * 74)
    cands = []
    for K in K_FRONTIER:
        f = frontier[K]
        cands.append((f"K={K} compressed", f['acc'], np.mean(f['bytes']),
                      f['kbpp'], f['ratio']))
    for (K, mode, th), v in fold_rows.items():
        cands.append((f"K={K} fold-{mode} th={th:.2f}", v['acc'], v['bytes'],
                      v['kbpp'], v['ratio']))
    for K, v in curve2.items():
        cands.append((f"K={K} compressed + {selected['name']}", v['acc'],
                      v['bytes'], v['kbpp'], v['ratio']))

    at_bar = sorted([c for c in cands if c[1] >= bar_acc[0]], key=lambda c: c[3])
    print(f"\ncheapest arms AT OR ABOVE the paired bar's ACC "
          f"{bar_acc[0]:.3f} (mean over s=0-4):")
    if at_bar:
        for lbl, a, b, k, rr in at_bar[:6]:
            print(f"  {lbl:<32} ACC {a:.3f}  {b / 1e3:6.1f}KB  "
                  f"{k:6.1f} KB/ACC-pt  {rr:5.2f}x the bar")
    else:
        print("  none")
    print(f"\nglobally cheapest per accuracy point (any accuracy):")
    for lbl, a, b, k, rr in sorted(cands, key=lambda c: c[3])[:6]:
        print(f"  {lbl:<32} ACC {a:.3f}  {b / 1e3:6.1f}KB  "
              f"{k:6.1f} KB/ACC-pt  {rr:5.2f}x the bar")

    print("\nan accounting caution the curve forces, stated plainly:")
    print("  KB/ACC-point is minimized at LOW capacity -- the organism beats "
          "the bar on the")
    print("  raw ratio at small K (see the list above) purely because "
          "accuracy per byte is")
    print("  steepest where accuracy is worst. The gate's phrase is "
          "'cost-effective AT EQUAL")
    print("  ACCURACY', so the constrained number -- cheapest arm at or "
          "above the bar's ACC")
    print("  -- is the only one that answers it. Every verdict below is the "
          "constrained one.")

    print("\nverdict vs the gate's cost branch:")
    print(f"  bar (paired, s=0-4)   {bar_kbpp[0]:6.1f} KB/ACC-pt at ACC "
          f"{bar_acc[0]:.3f}")
    print(f"  33g's K=160 arm       "
          f"{kbpp(ANCHOR_33G_K160_BYTES, ANCHOR_33G_K160_ACC):6.1f} "
          f"KB/ACC-pt (3.15x -- the number T1.9 opened against)")
    if at_bar:
        lbl, a, b, k, rr = at_bar[0]
        print(f"  33h best at bar       {k:6.1f} KB/ACC-pt  ({rr:.2f}x)  "
              f"-- {lbl}   [SELECTION SEEDS]")
    if held:
        # rule (d) applies to the held-out confirmation too, or the phase
        # would be doing on s=5-9 exactly what it refused to do on s=0-4:
        # bank a mean. The verdict point must be at/above the bar AND carry
        # its decoder gain on EVERY held-out seed.
        eligible = [K for K in held if held[K]['at_bar'] and held[K]['robust']]
        cheapest_mean = min([K for K in held if held[K]['at_bar']],
                            key=lambda K: held[K]['kbpp'], default=None)
        if cheapest_mean is not None and cheapest_mean not in eligible:
            cm = held[cheapest_mean]
            print(f"  33h held-out, mean only: {cm['kbpp']:6.1f} KB/ACC-pt "
                  f"({cm['ratio']:.2f}x) at K={cheapest_mean} -- NOT BANKED: "
                  f"its decoder gain\n{'':<26}sign-flips out of sample "
                  f"(min delta {cm['delta'].min():+.3f}), failing rule (d). "
                  f"Recorded, not claimed.")
        Kh = (min(eligible, key=lambda K: held[K]['kbpp']) if eligible
              else min(held, key=lambda K: held[K]['kbpp']))
        h = held[Kh]
        met_parity = h['ratio'] <= 1.0 and h['at_bar']
        met_2x = h['ratio'] <= BAR_RATIO_TARGET and h['at_bar']
        print(f"  33h HELD-OUT (s=5-9)  {h['kbpp']:6.1f} KB/ACC-pt  "
              f"({h['ratio']:.2f}x)  -- K={Kh} + {selected['name']}, ACC "
              f"{h['acc']:.3f} {'>=' if h['at_bar'] else '<'} bar "
              f"{bar_acc_h[0]:.3f}, gain holds on all 5 held-out seeds")
        print(f"\n  PARITY (1.0x): {'MET' if met_parity else 'NOT MET'}"
              f"   |   T1.9's <= {BAR_RATIO_TARGET:.0f}x fallback: "
              f"{'MET' if met_2x else 'NOT MET'}   (on the held-out seeds)")
        floor = (h['kbpp'], f"K={Kh} compressed + {selected['name']} "
                            f"(held out)", h['ratio'])
    elif at_bar:
        lbl, a, b, k, rr = at_bar[0]
        print(f"\n  PARITY (1.0x): {'MET' if rr <= 1.0 else 'NOT MET'}"
              f"   |   T1.9's <= {BAR_RATIO_TARGET:.0f}x fallback: "
              f"{'MET' if rr <= BAR_RATIO_TARGET else 'NOT MET'}")
        floor = (k, lbl, rr)
    else:
        best_any = min(cands, key=lambda c: c[3])
        print(f"  no arm reaches the bar's accuracy; cheapest overall "
              f"{best_any[3]:.1f} KB/ACC-pt at ACC {best_any[1]:.3f}")
        print(f"  PARITY: NOT MET   |   <= {BAR_RATIO_TARGET:.0f}x fallback: NOT MET")
        floor = (best_any[3], best_any[0], best_any[4])
    print(f"\n  MEASURED FLOOR: {floor[0]:.1f} KB/ACC-pt "
          f"({floor[2]:.2f}x the bar) -- {floor[1]}")

    # -- a SCOPE variation, not a lever, reported with its price ----------
    if held and bar_kbpp_h is not None:
        Kh = min(held, key=lambda K: held[K]['kbpp']) if not [
            K for K in held if held[K]['at_bar'] and held[K]['robust']] else \
            min([K for K in held if held[K]['at_bar'] and held[K]['robust']],
                key=lambda K: held[K]['kbpp'])
        h = held[Kh]
        gp = float(np.mean([compress(StoreView(run_arm(Kh, seed=s)['org']),
                                     spec=SPEC).p_bytes for s in SEEDS_HELD]))
        nog = kbpp(h['bytes'] - gp, h['acc'])
        print(f"\n  SCOPE VARIATION (reported because it is decision-relevant, "
              f"NOT claimed as a lever):")
        print(f"    this benchmark's readout never queries the transition "
              f"graph -- it scores label")
        print(f"    evidence over xi overlaps only (33g's integrity note). A "
              f"store that dropped the")
        print(f"    graph would cost {(h['bytes'] - gp) / 1e3:.1f}KB at ACC "
              f"{h['acc']:.3f} = {nog:.1f} KB/ACC-pt = "
              f"{nog / bar_kbpp_h[0]:.2f}x the bar, which")
        print(f"    {'MEETS' if nog / bar_kbpp_h[0] <= BAR_RATIO_TARGET else 'still misses'}"
              f" the <= {BAR_RATIO_TARGET:.0f}x fallback. That number is only "
              f"honest for a product that")
        print(f"    never reasons: the graph IS the logic layer (phase 30's "
              f"kstep/rollout/plan,")
        print(f"    phases 35/36's hierarchical routing), and every capability "
          f"the gate would be")
        print(f"    re-scoped TOWARD depends on it. Quoting "
              f"{nog / bar_kbpp_h[0]:.2f}x without this paragraph would be "
              f"the same")
        print(f"    measurement artifact 33g refused to bank on P pruning.")
    print(f"  reference: 33c's replay arm holds 0.913 at ~{REPLAY_BYTES / 1e3:.1f}KB "
          f"= {kbpp(REPLAY_BYTES, 0.913):.1f} KB/ACC-pt "
          f"({kbpp(REPLAY_BYTES, 0.913) / bar_kbpp[0]:.2f}x) and still tops the ladder.")
    print(f"\n  the reason, from the byte formulas (section 0's arithmetic):")
    print(f"    a field memory is COMPLEX, a prototype is REAL -> "
          f"{per_slot / per_proto:.2f}x per memory, before the graph;")
    print(f"    and the organism needs more memories than the bar for the "
          f"same accuracy.")
    print(f"    Those two factors multiply, and neither is a storage-"
          f"engineering target:")
    print(f"    the complex state IS the mechanism (phase is what carries "
          f"binding and")
    print(f"    sequence), and the slot count is set by the capacity curve "
          f"phase 33d measured.")
    print(f"\ndone in {time.time() - t_start:.1f}s")
