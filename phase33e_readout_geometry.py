"""
PHASE 33e (T1.6): READOUT GEOMETRY -- eval-side decoder comparison on the
phase-33c protocol. The 0.872 supervised prototype bar wins partly on
readout geometry (120 per-class prototypes, nearest-prototype decision),
not only on memory content; `LabelEvidenceReadout.predict`'s default is the
bluntest possible decoder (argmax-overlap slot -> majority label). This
phase measures whether richer EVAL-SIDE decoders over the SAME organism
state -- same slots, same evidence matrix, labels still entering at the
readout only -- claw back any of the 0.160 gate gap.

Decoders (label_readout.py, T1.6; every one bitwise non-mutating, pinned
by test_label_readout.py section 4):
  argmax        best-overlap evidenced slot -> majority label (the anchor;
                phase 33/33b/33c scoring, byte-for-byte)
  soft-mM       evidence-weighted soft vote over the top-M overlapping
                slots (label mass, not majority collapse)
  dist-mM       same routing, per-slot label DISTRIBUTIONS (count-
                normalized so a high-count slot cannot dominate on mass)
  calib-bB      softmax(beta * overlap) weights over ALL evidenced slots
                times per-slot distributions (distance-calibrated;
                beta -> inf = nearest-slot routing)

Protocol: phase 33c's two organism arms verbatim -- evict=0 (phase 33's
reproduction anchor) and evict=250 + readout with eviction invalidation
(the T1.4 budget arm). Same seeds, same stream construction (fresh rng
burns the split draw), same NumbaOrganism recipe, same task-accuracy
matrix scoring. Each arm is TRAINED ONCE; every decoder scores the same
trained state at every eval point (prediction never mutates, so this is
exact). No torch needed -- organism arms only.

Pre-registered predictions (written before the committed run):
  (a) Reproduction: the argmax rows must reproduce the 33c anchors
      EXACTLY -- evict=0 ACC 0.665 / FORG 0.200, evict=250 ACC 0.712 /
      FORG 0.169. If they don't, nothing else in the table is evidence.
  (b) AGENT_TARGETS T1.6(a): soft top-m recovers part of the gap --
      drifted/mixed slots carry label mass the majority collapse throws
      away (the ACC 0.25 -> 0.665 frozen-label history says slot-label
      ambiguity is real on this stream). Expected on the evict=250 arm
      especially (its slots are younger, mixed-era).
  (c) AGENT_TARGETS T1.6(b), the honest negative: if NO decoder moves ACC
      by > 0.02 on either arm, the memory content itself is the limit and
      the lever is T1.5 (capacity) / mechanism, not readout geometry --
      that null is worth pinning. Decision rule, fixed in advance: the
      claim threshold is best-decoder ACC minus argmax ACC > +0.02 on the
      SAME arm; anything inside +/-0.02 is null-band. FORG is reported
      but the gate axis (and the threshold) is ACC.
  (d) Default policy, fixed in advance: `argmax` stays the module default
      unless a decoder clears (c)'s threshold AND leaves the 33c anchors
      reproducible under the default path -- the anchors are scored with
      argmax regardless, so a default flip would require a new committed
      ladder row, not a silent re-score.
  (e) Robustness supplement (added after the seed-0 sweep showed budget-arm
      decoders clearing the threshold, BEFORE the supplement was run --
      best-of-14-configs on a single seed can be grid selection noise, and
      the standing noise-floor rule applies to a positive as much as to a
      null): re-run the budget arm under full-protocol reseeds (split,
      stream shuffles, organism all seeded s = 1..4; seed 0 is the
      committed protocol) and report dACC for TWO configs named in advance
      from the seed-0 sweep -- primary calib-b8, secondary dist-m2 --
      plus the per-seed best-of-grid as the selection-bias reference.
      Claim survives only if the named-config mean dACC stays > +0.02
      without sign flips; the honest negative is that the deltas collapse
      across seeds, in which case the seed-0 win is noise and the verdict
      reverts to the (c) null.

COMMITTED-RUN FINDINGS (2026-08-05; numpy 2.4.6, numba 0.66; environment
baseline harness 31/31 both backends + test_label_readout green before
the run). Prediction (a) held: both argmax rows EXACT (evict=0
0.665/0.200, evict=250 0.712/0.169). The seed-0 sweep looked like
prediction (b): on the budget arm five configs cleared +0.02 (calib-b8
+0.063 -> ACC 0.775, dist-m2 +0.051, dist-m3 +0.040, calib-b32 +0.035,
soft-m40 +0.023) while on the flooded evict=0 arm NOTHING beat argmax
(best alternative +0.000, wide routing destructive down to -0.443 at
calib-b2) -- geometry only ever helps where the slot population is
healthy/mixed-era, never where it is drift-dominated. But the
pre-registered robustness supplement (e) returned the honest negative:
under full-protocol reseeds the named configs go +0.063/+0.030/+0.088/
-0.031/+0.029 (calib-b8: mean +0.036, min -0.031) and +0.051/+0.002/
+0.053/-0.029/+0.058 (dist-m2: mean +0.027) -- positive mean but a sign
flip at seed 3, exactly the seed where argmax itself lands strongest
(0.813; per-seed range 0.692-0.813). The gain is ANTI-CORRELATED with
argmax's own strength: distribution decoders recover label mass when
the evidence lands messy and give it back when it lands clean. Per rule
(e) the claim does not survive; THE (c) NULL STANDS: no eval-side
decoder moves ACC by > 0.02 robustly. The memory content (K=40 = 4
slots/class vs the bar's 120 prototypes), not readout geometry, is the
limit -- the lever is T1.5/mechanism, as pre-registered. argmax stays
the default (prediction d). Pinned: test_label_readout.py section 5
asserts the committed evict=0 protocol null (no alternative decoder
beats argmax by > 0.02) under CI. Sub-findings for the record: the
seed-0 calib-b8 ACC gain even cost forgetting (FORG 0.190 vs argmax
0.169); exploratory grid x seed means top out at calib-b8 +0.036 /
dist-m3 +0.029 / dist-m2 +0.027 (descriptive only -- any future
confirmation needs its own pre-registration).
"""

import time

import numpy as np
from sklearn.datasets import load_digits

from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_numba import NumbaOrganism

d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(np.float32)
y = d.target.astype(int)
N = X.shape[1]
NORM = np.sqrt(N)
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
K = 40
EVICT = 250


def build_split(seed):
    """The phase-33 train/test split, seeded. seed=0 is the committed
    protocol byte-for-byte (same rng draw order as 33/33b/33c)."""
    r = np.random.default_rng(seed)
    perm = r.permutation(len(X))
    tr, te = perm[:1400], perm[1400:]
    return X[tr], y[tr], X[te], y[te]

# 33c anchors the argmax rows must land on (prediction a)
ANCHORS = {0: (0.665, 0.200), EVICT: (0.712, 0.169)}

# the decoder grid (name, predict kwargs); argmax first -- it is the anchor
DECODERS = [("argmax", dict(decoder="argmax"))]
DECODERS += [(f"soft-m{m}", dict(decoder="soft", m=m)) for m in (2, 3, 5, 10, 40)]
DECODERS += [(f"dist-m{m}", dict(decoder="dist", m=m)) for m in (2, 3, 5, 10, 40)]
DECODERS += [(f"calib-b{b:g}", dict(decoder="calib", m=None, beta=float(b)))
             for b in (2, 8, 32, 128)]

CLAIM_THRESH = 0.02     # pre-registered: |delta ACC| beyond this is a claim


def task_acc_matrix_to_metrics(A):
    """A[i, j] = accuracy on task j's classes after training task i."""
    T = len(TASKS)
    final = A[T - 1]
    acc = float(np.mean(final))
    forg = float(np.mean([np.max(A[:, j]) - final[j] for j in range(T - 1)]))
    return acc, forg


def eval_tasks(predict, Xte, yte):
    out = []
    for task in TASKS:
        m = np.isin(yte, task)
        out.append(float((predict(Xte[m]) == yte[m]).mean()))
    return out


def run_arm(evict, seed=0, hold=8, epochs=3):
    """Phase 33c's organism arm verbatim (stream, recipe, readout
    bookkeeping), trained once; every decoder scores every eval point.
    seed=0 is the committed protocol; other seeds are full-protocol
    reseeds (split, shuffles, organism) for the robustness supplement."""
    Xtr, ytr, Xte, yte = build_split(seed)
    r = np.random.default_rng(seed)
    r.permutation(len(X))                    # burn the split draw (33b/33c)
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=seed)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    A = {name: np.zeros((len(TASKS), len(TASKS))) for name, _ in DECODERS}
    t0 = time.time()
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
        for name, kw in DECODERS:
            A[name][ti] = eval_tasks(
                lambda Xe: readout.predict(org, Xe, **kw), Xte, yte)
    return A, time.time() - t0


if __name__ == "__main__":
    print("PHASE 33e (T1.6): readout geometry -- eval-side decoders, 33c protocol\n")
    verdict_rows = []
    for evict in (0, EVICT):
        arm = f"evict={evict}"
        print(f"arm {arm} ({'33 anchor' if evict == 0 else 'T1.4 budget arm'}):")
        A, t = run_arm(evict)
        acc0, forg0 = task_acc_matrix_to_metrics(A["argmax"])
        want_acc, want_forg = ANCHORS[evict]
        ok = abs(acc0 - want_acc) < 5e-4 and abs(forg0 - want_forg) < 5e-4
        print(f"  {'argmax':<10} ACC {acc0:.3f}  FORG {forg0:.3f}   "
              f"(anchor {want_acc:.3f}/{want_forg:.3f}: "
              f"{'REPRODUCED' if ok else 'MISMATCH -- table is not evidence'})")
        best_name, best_acc = "argmax", acc0
        for name, _ in DECODERS[1:]:
            acc, forg = task_acc_matrix_to_metrics(A[name])
            flag = ""
            if acc - acc0 > CLAIM_THRESH:
                flag = "  << clears the +0.02 claim threshold"
            elif acc0 - acc > CLAIM_THRESH:
                flag = "  (destructive)"
            print(f"  {name:<10} ACC {acc:.3f}  FORG {forg:.3f}   "
                  f"dACC {acc - acc0:+.3f}{flag}")
            if acc > best_acc:
                best_name, best_acc = name, acc
        verdict_rows.append((arm, acc0, best_name, best_acc, ok))
        print(f"  (trained once, {len(DECODERS)} decoders scored; {t:.1f}s)\n")

    print("verdict (pre-registered decision rule: best - argmax > +0.02 on")
    print("the same arm claims a geometry recovery; inside +/-0.02 is null):")
    any_claim = False
    for arm, acc0, best_name, best_acc, ok in verdict_rows:
        delta = best_acc - acc0
        line = (f"  {arm:<11} argmax {acc0:.3f}; best alternative {best_name} "
                f"{best_acc:.3f} (dACC {delta:+.3f})")
        if not ok:
            line += "  [anchor MISMATCH -- excluded from any claim]"
        print(line)
        any_claim |= ok and delta > CLAIM_THRESH
    if any_claim:
        print("  -> a decoder clears the threshold: readout geometry carries")
        print("     real gap; per prediction (d) a default flip still needs a")
        print("     committed ladder row with the anchors re-scored under a flag.")
    else:
        print("  -> THE NULL HOLDS (prediction c): no eval-side decoder moves")
        print("     ACC by > 0.02 on either arm. The memory content, not the")
        print("     readout geometry, is the limit; the lever is T1.5/mechanism.")
        print("     argmax stays the default. Pin this null (test_label_readout")
        print("     section 5) so it stays measured.")

    if any_claim:
        print("\nrobustness supplement (pre-registered prediction e: named "
              "configs\ncalib-b8 / dist-m2, budget arm, full-protocol reseeds; "
              "per-seed\nbest-of-grid shown only as the selection-bias "
              "reference):")
        named = ["calib-b8", "dist-m2"]
        all_d = {n: [] for n, _ in DECODERS[1:]}
        arg_accs = []
        for s in (0, 1, 2, 3, 4):
            A, _ = run_arm(EVICT, seed=s)
            acc0, _ = task_acc_matrix_to_metrics(A["argmax"])
            arg_accs.append(acc0)
            accs = {n: task_acc_matrix_to_metrics(A[n])[0]
                    for n, _ in DECODERS[1:]}
            best_n = max(accs, key=accs.get)
            for n in all_d:
                all_d[n].append(accs[n] - acc0)
            print(f"  seed {s}: argmax {acc0:.3f}  "
                  + "  ".join(f"{n} {accs[n]:.3f} ({accs[n] - acc0:+.3f})"
                              for n in named)
                  + f"  [best-of-grid: {best_n} {accs[best_n] - acc0:+.3f}]")
        print("  named-config mean dACC over 5 seeds: "
              + "  ".join(f"{n} {np.mean(all_d[n]):+.3f} "
                          f"(min {np.min(all_d[n]):+.3f})" for n in named))
        survives = []
        for n in named:
            m_, lo = float(np.mean(all_d[n])), float(np.min(all_d[n]))
            if m_ > CLAIM_THRESH and lo > 0:
                survives.append(n)
                print(f"  -> {n}: SURVIVES (mean > +0.02, no sign flip)")
            else:
                print(f"  -> {n}: does NOT survive reseeding "
                      f"(mean {m_:+.3f}, min {lo:+.3f})")
        top3 = sorted(all_d, key=lambda n: -np.mean(all_d[n]))[:3]
        print("  exploratory (grid x seeds selection -- descriptive only, NO"
              "\n  claim; a future confirmation would need its own "
              "pre-registration):")
        for n in top3:
            print(f"    {n:<10} mean {np.mean(all_d[n]):+.3f}  "
                  f"per-seed {[f'{v:+.3f}' for v in all_d[n]]}")

        print("\nFINAL verdict (pre-registered rule e):")
        if survives:
            print(f"  {', '.join(survives)} survive full-protocol reseeding --")
            print("  readout geometry carries real, robust gap on the budget")
            print("  arm; a default flip still requires a committed ladder row")
            print("  (prediction d).")
        else:
            print("  the seed-0 geometry wins are SEED-CONDITIONAL: positive")
            print("  mean but sign flips under reseeding (the gain is largest")
            print("  where argmax lands weak and vanishes where it lands")
            print("  strong -- argmax per-seed range "
                  f"{min(arg_accs):.3f}-{max(arg_accs):.3f}). Per the")
            print("  pre-registered survival rule the (c) NULL stands: no")
            print("  eval-side decoder moves ACC by > 0.02 ROBUSTLY. The")
            print("  memory content, not readout geometry, is the limit; the")
            print("  lever is T1.5/mechanism. argmax stays the default; the")
            print("  evict=0 committed-protocol null is pinned in")
            print("  test_label_readout.py section 5.")
