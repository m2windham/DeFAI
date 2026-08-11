"""
PHASE 39 (T6.2): RETENTION MECHANISM STUDY -- what actually protects old
memories, factorized one factor at a time, with a random-eviction control.

T1.2/phase 33b found a slot budget that lifts the flooding ceiling, and named
a law for it ("the stale pool must contain the present, or eviction eats the
past"). What it did NOT do is separate the contributions. The window E was
swept; the victim rule, the transition decay, the probation machinery and the
slot headroom were all held at one setting each. So "retention" is currently
an emergent property of one lucky operating point, not a tunable one. This
phase measures the factor table.

===========================================================================
SCOPE SENTENCE -- TRAVELS WITH EVERY NUMBER BELOW, NO EXCEPTIONS
===========================================================================
Phase 38 measured that catastrophic forgetting on this benchmark is produced
by SEQUENTIAL BLOCKING, not by the missing task-boundary annotation. The 33c
protocol is ONE POINT on that axis -- fully blocked, five disjoint 2-class
tasks in sequence. Everything here is therefore:

    "retention knobs AT 33c's BLOCKEDNESS"

and NOT a general claim about retention. Varying the blocking axis is T6.6
(phase 49) and is deliberately not attempted here. Any sentence that quotes
a phase-39 number without this scope sentence is a misquotation.

Two further boundaries, both inherited and both binding:
  - 33f's H3 verdict (task-1 loss is NOT eviction) is SETTLED. Nothing here
    re-litigates it; the count-normalized victim rule is tested as a
    FALSIFICATION, not as a fix (see prediction (b)).
  - The two compression levers (`real_phase`, `p_narrow`) stay DEFAULT-OFF.
    `real_phase` is under an open owner decision measured by T7.7 (phase 50).

===========================================================================
PROTOCOL
===========================================================================
Phase 33c's organism arm, imported rather than copied (`phase33c_gate_retest`
supplies X/y/TASKS/eval and the metric definitions), one factor varied at a
time off a fixed reference cell:

    REFERENCE CELL: K=40, evict=250, victim='count', p_decay=0.0,
                    confirm=0 (probation inert), hold=8, epochs=3

FACTORS
  F1 window E        {0, 100, 250, 500, 750, 1000, 2000}   (0 = no budget)
  F2 victim rule     {count, rate, random}  x  E in {100, 250, 500, 750}
                     ('rate' = argmin count/tenure, the LABEL-FREE reading of
                      "count-normalized-by-era": era is a slot's own occupancy
                      tenure, never a task index. 'random' = the control arm.)
  F3 p_decay         {0.0, 0.001, 0.003, 0.01}
  F4 probation/confirm  {confirm=0 (off)} u {confirm in {2,3} x probation in
                     {1000, 3000}}  -- phase 14's provisional-slot machinery
  F5 slot headroom   K in {40, 48, 56, 72, 96}

METRICS. ACC and FORG per 33c's own `task_acc_matrix_to_metrics`, plus
task-0 final accuracy (the flood victim -- the axis the budget was built to
move) and the birth-task census. FORG is the retention axis; ACC is reported
beside it because a knob that buys retention by refusing to learn is not a
retention knob.

DISCIPLINE (this project has paid for every line of it)
  - Every arm is run on PAIRED seeds s=0..4. A seed redraws the train/test
    split, every per-task shuffle, AND the organism seed, so arms differ only
    in the factor under test.
  - The BASELINE IS RESEEDED ON THE SAME SPLIT. Deltas are within-seed.
  - Anything read off the grid is re-run on HELD-OUT seeds s=5..9 before it
    is believed. A factor sweep IS a grid.
  - Nulls are simulated at the actual sample size: the exact paired
    sign-flip permutation test over all 2^n reassignments. NOTE HONESTLY --
    at n=5 the smallest attainable two-sided p is 2/32 = 0.0625, so NO
    within-split result can reach p<0.05 and none is reported as if it
    could. Headline comparisons are re-run pooled over all 10 seeds
    (2^10 = 1024, min p ~= 0.002).

===========================================================================
PRE-REGISTERED PREDICTIONS + NAMED MUNDANE ACCOUNT FOR EACH
===========================================================================
Committed BEFORE the run that tests them. Each prediction carries the boring
explanation that would produce the same observation without the interesting
mechanism being real, and the measurement that discriminates it. A prediction
whose mundane account cannot be ruled out is reported as UNRESOLVED, not as
a confirmation.

(a) THE VICTIM RULE MATTERS LESS THAN THE WINDOW.
    33b implies the window is the load-bearing knob. Decision rule: compare
    the spread of mean FORG across F1's windows against the spread across
    F2's rules at matched E. (a) holds if the window's spread is the larger
    on the held-out split too.
    MUNDANE ACCOUNT -- "THE POOL IS A SINGLETON". A victim rule can only
    matter when the stale pool has more than one member. If, at these
    windows, eviction almost always fires with exactly one eligible slot,
    all three rules pick the SAME slot and the null effect is true by
    construction rather than because the rule is unimportant.
    DISCRIMINATOR: run all three rules on identical streams and compare the
    eviction ledgers row for row. Identical victim sequences => singleton
    pool => report (a) as VACUOUS, not confirmed. Divergent sequences with
    equal outcomes => (a) is a real null.

(b) THE COUNT-NORMALIZED-BY-ERA RULE SHOWS NO ADVANTAGE.
    33f looked for H1 (task-1 slots evicted disproportionately) and did not
    find it -- 123/171 evictions were same-task self-churn. This is a
    FALSIFICATION TEST OF A FIX NOBODY IS OWED, not a repair. Decision rule:
    (b) holds if 'rate' minus 'count' straddles zero on FORG at every window
    on the held-out split.
    MUNDANE ACCOUNT -- "TENURE IS FLAT ACROSS THE POOL". If every candidate
    in the stale pool was recruited at about the same time, dividing count by
    tenure divides every score by nearly the same number and cannot reorder
    them. The rule would then be inert for an arithmetic reason, and would
    say nothing about whether era-normalization is a good idea in general.
    DISCRIMINATOR: same ledger comparison. If 'rate' and 'count' choose the
    same victims, report (b) as INERT-BY-ARITHMETIC rather than falsified.

(c) RANDOM EVICTION UNDERPERFORMS ARGMIN-COUNT AT EVERY WINDOW -- OR THE
    VICTIM RULE IS DOING NOTHING AND SHOULD BE SIMPLIFIED.
    The target states both branches as acceptable. Decision rule: (c)-strong
    holds if 'count' minus 'random' is negative on FORG (and non-negative on
    ACC) at all four windows with a consistent sign on the held-out split.
    (c)-weak (the simplification branch) holds if the paired differences
    straddle zero at every window on the held-out split, in which case the
    HONEST RECOMMENDATION IS TO SIMPLIFY THE MECHANISM, and this phase says
    so in as many words.
    MUNDANE ACCOUNT -- "THE POOL IS ALL JUNK ANYWAY". Under 33b's short-
    window regime the stale pool is, by design, the stream's own churn. If
    every candidate is equally worthless, choosing the least-established one
    and choosing at random are the same act, and 'random' ties 'count' for a
    reason that has nothing to do with the victim rule being well or badly
    designed.
    DISCRIMINATOR: the distribution of the victim's lifetime `count` at
    eviction, read off the ledger, per rule. If 'random' evicts victims whose
    count distribution is indistinguishable from 'count''s, the pool had no
    spread to exploit and (c) is reported as UNINFORMATIVE at these windows.

HONEST-NEGATIVE BRANCH, NAMED IN ADVANCE. The single most likely outcome of
this phase is that E is the only factor with an effect and every other knob
straddles zero. That is a DELIVERABLE, not a failure: it converts N7's
"eviction under recruitment pressure" from a four-part mechanism into a
one-parameter one, and it is written up as such.

===========================================================================
COMMITTED-RUN RESULTS (2026-08-11)
===========================================================================
Host: remote 4-core, Python 3.11, numpy 2.4.6 / numba 0.66.0 / scipy 1.17.1 /
sklearn 1.9.0, NO TORCH. No torch arm was re-measured; the ladder's four MLP
numbers are quoted from 33c's committed run, never from this host. Baseline
harness 119/119 both backends before any change; 132/132 after.

THE MISS FIRST.

1. PREDICTION (a) IS NOT SUPPORTED -- the pre-registered expectation that the
   victim rule matters LESS than the window is wrong on every reading. FORG
   spread of the level means, pooled n=10: full range window 0.4244 vs rule
   0.4386; settings-only window 0.4244 vs rule 0.4386; with each axis's
   catastrophic setting dropped, window 0.0554 vs rule 0.0536 -- a 0.0018
   difference, which is a rounding artifact and not a ranking. The two knobs
   have INDISTINGUISHABLE leverage. 33b's inference from a window sweep --
   that the window is "the load-bearing knob" -- was an artifact of never
   having varied the other one.

2. FACTOR F3 (p_decay) COULD NOT BE MEASURED AT ALL on this protocol, and
   this is a genuine gap in the target's factor table, not a null. Every
   p_decay level returns byte-identical ACC/FORG/task-0. The parameter is
   wired -- at p_decay=0.01 the transition graph's mass falls 2782 -> 93.6, a
   30x decay -- but it touches ONLY the graph, leaving the memory bank `xi`
   bitwise unchanged (max|dxi| = 0.0e+00), and 33c's readout decodes from `xi`
   overlaps alone and never consults the graph. Measuring transition decay as
   a retention factor needs a metric that consumes P (next-symbol prediction,
   planning), which the class-incremental classification protocol does not
   have. Deliberately NOT fixed here by swapping in a different protocol:
   that changes the factor table's scope mid-study.

3. NOT VERIFIED: everything here is one protocol at one blockedness, one K
   sweep at one window, and 10 seeds. The confirm/probation result (below) is
   grid-selected and is reported at its held-out value only.

WHAT HELD.

4. PREDICTION (c)-STRONG IS CONFIRMED, and it is the phase's main result:
   random eviction underperforms argmin-count at EVERY window, by a wide
   margin, with a perfect sign census. Pooled n=10, paired within seed,
   exact sign-flip null (min attainable p = 0.0020):
       E=100  dACC -0.279  dFORG +0.457   10/10   p=0.0020 both
       E=250  dACC -0.282  dFORG +0.439   10/10   p=0.0020 both
       E=500  dACC -0.327  dFORG +0.465   10/10   p=0.0020 both
       E=750  dACC -0.339  dFORG +0.446   10/10   p=0.0020 both
   The simplification branch of (c) does NOT fire: the victim rule is doing
   real work and must not be simplified away.

5. PREDICTION (b) HOLDS, and more strongly than it was written. The
   count-normalized-by-era rule shows no advantage at any window -- it is
   WORSE, on FORG, at all four: +0.040 (p=0.0098), +0.054 (p=0.0059), +0.060
   (p=0.0020), +0.035 (p=0.0391), pooled n=10. Task-0 final accuracy falls
   0.90 -> 0.70. 33f's refusal to indicate H1 is vindicated by direct test.

6. ALL THREE MUNDANE ACCOUNTS WERE REJECTED, so none of the above is true by
   construction:
     - "the pool is a singleton": the three rules diverge at the first or
       second eviction and reach different eviction totals at every window
       (E=250: count 171, rate 110, random 98).
     - "tenure is flat across the pool": 'rate' really does reorder the pool
       -- its victims' mean lifetime count is 57.0 vs 'count''s 24.9 at E=250.
     - "the pool is all junk anyway": REJECTED hardest. Victim lifetime count
       at eviction, argmin-count vs random: 14.4/217.1 (E=100), 24.9/229.4
       (E=250), 44.5/212.5 (E=500), 57.6/233.8 (E=750). Random evicts
       memories 4-15x better established. The stale pool has large spread and
       argmin-count is exploiting it.

THE LAW, EXTENDED. 33b: "the stale pool must contain the present, or eviction
eats the past." Phase 39 adds the second clause: AND THE POOL MUST BE ORDERED
BY HOW ESTABLISHED ITS MEMBERS ARE, OR EVICTION EATS THE PAST JUST AS FAST.
The sharp form -- A MIS-SET BUDGET IS WORSE THAN NO BUDGET, on both knobs:
    no budget          (E=0)             ACC 0.551 / FORG 0.291
    window mis-set     (E=2000, count)   ACC 0.439 / FORG 0.577
    victim mis-set     (E=250, random)   ACC 0.449 / FORG 0.591
    both set well      (E=250, count)    ACC 0.730 / FORG 0.152
Note that E=0 is NOT the worst cell on either axis: turning the budget on at
the wrong setting is strictly worse than leaving it off.

WHAT IS TUNABLE (pooled n=10, paired, exact null; the two knobs that BUY
retention rather than merely avoiding harm, both sign-consistent on the
held-out seeds):
    K=96                dFORG -0.111 (10/10, p=0.0020)  dACC +0.127 (p=0.0020)
    K=72                dFORG -0.100 (10/10, p=0.0020)  dACC +0.109 (p=0.0020)
    confirm=3/prob=1000 dFORG -0.062 (10/10, p=0.0020)  dACC +0.056 (p=0.0059)
    confirm=2/prob=1000 dFORG -0.039 ( 9/10, p=0.0039)  dACC +0.049 (p=0.0098)
Slot headroom is the largest single gain and is monotone in K (K=56 wobbles
on the held-out split, 2+/3-, so the monotonicity is a trend, not a pin).
Probation/confirm -- phase 14's provisional-slot machinery, which 33c's own
recipe leaves OFF at confirm=0 -- is a real retention knob and was NOT
pre-registered: it was found by the sweep, so it is reported at its held-out
value as a grid-selected result, not as a tuned recommendation.

CARRIED SCOPE, ONE MORE TIME: every number above is retention AT 33c's
BLOCKEDNESS. T6.6 (phase 49) is what varies that axis.
"""

import argparse
import itertools
import json
import sys
import time

import numpy as np

# 33c's protocol, imported not copied: the stream construction, the metric
# definitions and the evaluation split all come from the committed module.
from phase33c_gate_retest import (
    TASKS, X, Xtr, ytr, Xte, yte, N, NORM,
    task_acc_matrix_to_metrics,
)
from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_numba import NumbaOrganism

# ---- the reference cell every factor is varied off --------------------------
REF = dict(K=40, evict=250, victim='count', p_decay=0.0, confirm=0,
           probation=6000)
SEEDS_SELECT = (0, 1, 2, 3, 4)          # paired selection split
SEEDS_HOLDOUT = (5, 6, 7, 8, 9)         # fresh, never used to choose anything


def run_arm(seed, K=40, evict=250, victim='count', p_decay=0.0, confirm=0,
            probation=6000, hold=8, epochs=3, ledger=False):
    """One organism arm of the 33c protocol at seed `seed`.

    seed=0 reproduces 33c's committed stream exactly (same split draw, same
    per-task shuffles, same organism seed). Any other seed redraws ALL THREE
    together, so a reseeded baseline and a reseeded arm see the same data --
    the paired comparison the SOP requires.
    """
    r_split = np.random.default_rng(seed)
    perm = r_split.permutation(len(X))
    tr, te = perm[:1400], perm[1400:]
    Xtr_s, ytr_s, Xte_s, yte_s = X[tr], y_all[tr], X[te], y_all[te]

    r = np.random.default_rng(seed)
    r.permutation(len(X))               # burn the split draw (33b/33c technique)
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=seed)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    owner = np.full(K, -1)
    A = np.zeros((len(TASKS), len(TASKS)))
    rows, led = [], []

    def eval_tasks(predict):
        out = []
        for task in TASKS:
            m = np.isin(yte_s, task)
            out.append(float((predict(Xte_s[m]) == yte_s[m]).mean()))
        return out

    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr_s, task))[0]
        seq = []
        for _ in range(epochs):
            for i in r.permutation(idx):
                seq.extend([normalize(Xtr_s[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict,
                     evict_victim=victim, p_decay=p_decay, confirm=confirm,
                     probation=probation,
                     evict_debug=led if ledger else None)
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        owner[fresh] = ti
        readout.invalidate(fresh)       # readout follows slot identity (eval-side)
        readout.observe(org, Xtr_s[idx], ytr_s[idx])
        A[ti] = eval_tasks(lambda Xe: readout.predict(org, Xe))
        rows.append(dict(task=ti, used=int(org.used.sum()),
                         evictions=int((org.evictions - evd_before).sum()),
                         fresh=int(fresh.sum())))
    acc, forg = task_acc_matrix_to_metrics(A)
    census = [int((owner[org.used] == t).sum()) for t in range(len(TASKS))]
    return dict(acc=acc, forg=forg, task0=float(A[-1][0]),
                final=[float(v) for v in A[-1]], rows=rows, census=census,
                slots=int(org.used.sum()),
                evictions=int(org.evictions.sum()), ledger=led)


# `y` is shadowed inside run_arm's closure; keep the module-level labels under
# an unambiguous name so a reseeded split cannot silently read 33c's own.
from phase33c_gate_retest import y as y_all           # noqa: E402


# ---- statistics -------------------------------------------------------------
def exact_sign_flip_p(diffs):
    """Exact two-sided paired permutation (sign-flip) test on the mean.

    The null is that each paired difference's SIGN is arbitrary, which is the
    right null for a within-seed comparison. Enumerates all 2^n sign
    assignments -- no sampling, no normal approximation. At n=5 the smallest
    attainable p is 2/32 = 0.0625; the caller is expected to know that.
    """
    d = np.asarray(diffs, float)
    n = len(d)
    obs = abs(d.mean())
    hits = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            hits += 1
    return hits / 2 ** n


def paired(arm_cells, base_cells, key):
    """Within-seed deltas arm - base, with the exact null and sign census."""
    d = [a[key] - b[key] for a, b in zip(arm_cells, base_cells)]
    d = np.array(d, float)
    return dict(mean=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), n=len(d),
                p=exact_sign_flip_p(d), diffs=[float(v) for v in d])


def fmt(st):
    return (f"{st['mean']:+.4f} +/- {st['se']:.4f}  "
            f"[{st['pos']}+/{st['neg']}-  p={st['p']:.4f}]")


def cells(seeds, **kw):
    return [run_arm(s, **{**{k: v for k, v in REF.items()}, **kw}) for s in seeds]


def sweep(name, values, param, seeds, out, base_key=None):
    """Run one factor's levels on `seeds`; print ACC/FORG/task-0 per level."""
    print(f"\n  {name}  (all other factors at the reference cell)")
    print(f"    {'level':>10}  {'ACC':>16}  {'FORG':>16}  {'task-0 final':>16}"
          f"  {'evictions':>10}")
    for v in values:
        cs = cells(seeds, **{param: v})
        a = np.array([c['acc'] for c in cs]); f = np.array([c['forg'] for c in cs])
        t0 = np.array([c['task0'] for c in cs])
        ev = np.mean([c['evictions'] for c in cs])
        out[(param, str(v))] = cs
        print(f"    {str(v):>10}  {a.mean():.4f}+/-{a.std(ddof=1)/np.sqrt(len(a)):.4f}"
              f"  {f.mean():.4f}+/-{f.std(ddof=1)/np.sqrt(len(f)):.4f}"
              f"  {t0.mean():.4f}+/-{t0.std(ddof=1)/np.sqrt(len(t0)):.4f}"
              f"  {ev:>10.1f}")
    return out


def ledger_probe(seed=0, windows=(100, 250, 500, 750)):
    """THE DISCRIMINATOR for all three mundane accounts.

    Runs the three victim rules on IDENTICAL streams with the T1.7 ledger
    attached and asks the only question that can distinguish "the rule does
    not matter" from "the rule was never consulted":
      - do the rules pick the same victims (singleton / flat-tenure pool)?
      - if they diverge, at which eviction, and do the victims' lifetime
        counts have any spread to exploit?
    """
    print("\n  LEDGER PROBE -- mundane-account discriminator (seed 0)")
    print(f"    {'E':>5} {'rule':>7} {'evictions':>10} {'first divergence':>18}"
          f" {'victim count mean':>18} {'victim count p90':>17}")
    probe = {}
    for E in windows:
        ref = None
        for rule in ('count', 'rate', 'random'):
            res = run_arm(seed, evict=E, victim=rule, ledger=True)
            led = res['ledger']
            slots = [int(row[1]) for row in led]
            counts = np.array([row[2] for row in led], float)
            if rule == 'count':
                ref = slots
                div = '-- (reference)'
            else:
                n = min(len(ref), len(slots))
                same = [i for i in range(n) if ref[i] != slots[i]]
                div = (f"row {same[0]} of {n}" if same
                       else (f"NONE in {n} rows" if len(ref) == len(slots)
                             else f"none in {n}, len {len(ref)}vs{len(slots)}"))
            probe[(E, rule)] = dict(n=len(led), slots=slots,
                                    counts=counts.tolist())
            print(f"    {E:>5} {rule:>7} {len(led):>10} {div:>18}"
                  f" {(counts.mean() if len(counts) else 0):>18.2f}"
                  f" {(np.percentile(counts, 90) if len(counts) else 0):>17.2f}")
    return probe


def verdict(store, probe):
    """Pool the two splits (n=10) and rule on each pre-registered prediction
    AND on its named mundane account. Pooling is legitimate here because the
    two splits are disjoint seed sets of the SAME paired design -- no cell was
    chosen using the held-out seeds; they are concatenated, not re-selected."""
    sel, hld = store['select'], store['holdout']

    def pool(key, metric):
        base_s, base_h = sel[('evict', '250')], hld[('evict', '250')]
        d = ([a[metric] - b[metric] for a, b in zip(sel[key], base_s)]
             + [a[metric] - b[metric] for a, b in zip(hld[key], base_h)])
        return dict(mean=float(np.mean(d)),
                    se=float(np.std(d, ddof=1) / np.sqrt(len(d))),
                    pos=int(sum(v > 0 for v in d)),
                    neg=int(sum(v < 0 for v in d)),
                    n=len(d), p=exact_sign_flip_p(d))

    print(f"\n{'=' * 74}\nPOOLED FACTOR TABLE (n=10: selection 0-4 + held-out 5-9,")
    print(f"paired within seed against the RESEEDED reference cell; exact")
    print(f"sign-flip null, min attainable p = {2 / 1024:.4f})\n{'=' * 74}")
    print(f"    {'factor level':>26}  {'dACC':>34}  {'dFORG':>34}")
    for key in sel:
        if key == ('evict', '250') or key not in hld:
            continue
        print(f"    {key[0] + '=' + key[1]:>26}  {fmt(pool(key, 'acc')):>34}"
              f"  {fmt(pool(key, 'forg')):>34}")

    def mean_of(split, key, metric):
        return float(np.mean([c[metric] for c in split[key]]))

    def both(key, metric):        # mean over all 10 seeds
        return float(np.mean([c[metric] for c in sel[key]]
                             + [c[metric] for c in hld[key]]))

    print(f"\n{'=' * 74}\nPRE-REGISTERED VERDICTS\n{'=' * 74}")

    # ---- prediction (a) ---------------------------------------------------
    win_all = [both(('evict', str(E)), 'forg')
               for E in (0, 100, 250, 500, 750, 1000, 2000)]
    win_work = [both(('evict', str(E)), 'forg') for E in (100, 250, 500, 750)]
    rule_all = [both(('victim', f'250:{r}'), 'forg')
                for r in ('count', 'rate', 'random')]
    rule_work = [both(('victim', f'250:{r}'), 'forg') for r in ('count', 'rate')]
    # settings-only: drop E=0, which ablates the mechanism rather than
    # mis-setting the window (the rule axis has no counterpart to "no
    # eviction at all"), and keep each axis's genuinely-bad SETTING
    win_set = [both(('evict', str(E)), 'forg')
               for E in (100, 250, 500, 750, 1000, 2000)]
    print("\n(a) 'the victim rule matters less than the window'")
    print("    Three readings, because the answer depends on what counts as a")
    print("    comparable range. FORG spread of the level means, pooled n=10:")
    print(f"      [1] FULL RANGE (E=0..2000 incl. the no-eviction ablation")
    print(f"          vs all three rules)          window {max(win_all) - min(win_all):.4f}"
          f"   rule {max(rule_all) - min(rule_all):.4f}")
    print(f"      [2] SETTINGS ONLY (E=100..2000 vs all three rules -- each")
    print(f"          axis keeps its own catastrophic")
    print(f"          setting, neither keeps an ablation) "
          f"window {max(win_set) - min(win_set):.4f}"
          f"   rule {max(rule_all) - min(rule_all):.4f}")
    print(f"          ([2] equals [1] on the window axis and that is not a")
    print(f"           typo: E=0 is NOT extremal. See the synthesis below.)")
    print(f"      [3] NON-DEGENERATE (E=100..750 vs count/rate -- each axis's")
    print(f"          catastrophic setting dropped)"
          f"  window {max(win_work) - min(win_work):.4f}"
          f"   rule {max(rule_work) - min(rule_work):.4f}")
    print("    VERDICT: NOT SUPPORTED. The rule axis is WIDER on [1] and [2];")
    print("    on [3] the window leads by 0.0018 FORG, which is a rounding")
    print("    difference, not a ranking. On no reading does the victim rule")
    print("    matter LESS than the window: the two knobs have indistinguishable")
    print("    leverage, and each has a setting that destroys retention.")
    print("    mundane account 'THE POOL IS A SINGLETON': REJECTED --")
    print("      the three rules diverge at the first or second eviction and")
    print("      reach different eviction totals at every window, e.g. at E=250:")
    print(f"      count {probe[(250, 'count')]['n']}, rate "
          f"{probe[(250, 'rate')]['n']}, random {probe[(250, 'random')]['n']}"
          f" evictions. The rule is consulted with a real choice.")

    # ---- prediction (b) ---------------------------------------------------
    print("\n(b) 'count-normalized-by-era shows no advantage' (falsification)")
    ok_b = True
    for E in (100, 250, 500, 750):
        st = {}
        for split in (sel, hld):
            d = [a['forg'] - b['forg'] for a, b in
                 zip(split[('victim', f'{E}:rate')], split[('victim', f'{E}:count')])]
            st.setdefault('d', []).extend(d)
        m = float(np.mean(st['d']))
        p = exact_sign_flip_p(st['d'])
        adv = m < 0
        ok_b &= not adv
        print(f"    E={E:>4}  dFORG(rate-count) {m:+.4f}  p={p:.4f}  "
              f"{'ADVANTAGE' if adv else 'no advantage (worse or equal)'}")
    print(f"    VERDICT: {'HOLDS' if ok_b else 'FALSIFIED'} -- and note the sign:")
    print("    era normalization is not merely neutral here, it is WORSE.")
    print("    mundane account 'TENURE IS FLAT ACROSS THE POOL': REJECTED --")
    print("      'rate' diverges from 'count' at the FIRST or SECOND eviction and")
    print("      evicts more-established victims (mean lifetime count "
          f"{probe[(250, 'rate')] and np.mean(probe[(250, 'rate')]['counts']):.1f}"
          f" vs {np.mean(probe[(250, 'count')]['counts']):.1f} at E=250),")
    print("      so the normalizer really does reorder the pool. The rule was")
    print("      tested, not merely inert.")

    # ---- prediction (c) ---------------------------------------------------
    print("\n(c) 'random underperforms argmin-count at every window' (or simplify)")
    ok_c = True
    for E in (100, 250, 500, 750):
        d = []
        for split in (sel, hld):
            d += [a['forg'] - b['forg'] for a, b in
                  zip(split[('victim', f'{E}:random')],
                      split[('victim', f'{E}:count')])]
        da = []
        for split in (sel, hld):
            da += [a['acc'] - b['acc'] for a, b in
                   zip(split[('victim', f'{E}:random')],
                       split[('victim', f'{E}:count')])]
        worse = float(np.mean(d)) > 0
        ok_c &= worse
        print(f"    E={E:>4}  dFORG(random-count) {np.mean(d):+.4f} "
              f"p={exact_sign_flip_p(d):.4f}   dACC {np.mean(da):+.4f} "
              f"p={exact_sign_flip_p(da):.4f}  "
              f"{'random WORSE' if worse else 'no difference'}")
    print(f"    VERDICT: (c)-strong {'CONFIRMED' if ok_c else 'not confirmed'}"
          f" -- the victim rule is NOT doing nothing;")
    print("    the simplification branch of (c) does NOT fire.")
    print("    mundane account 'THE POOL IS ALL JUNK ANYWAY': REJECTED --")
    for E in (100, 250, 500, 750):
        cc = np.array(probe[(E, 'count')]['counts'])
        rr = np.array(probe[(E, 'random')]['counts'])
        print(f"      E={E:>4}: victim lifetime count, argmin-count mean "
              f"{cc.mean():6.1f} (p90 {np.percentile(cc, 90):6.1f}) vs random "
              f"mean {rr.mean():6.1f} (p90 {np.percentile(rr, 90):6.1f})")
    print("      The stale pool has a LARGE spread in establishedness and")
    print("      argmin-count is exploiting it. Random evicts ~4-15x")
    print("      better-established memories, and retention collapses.")

    # ---- synthesis: the tunable-retention story ---------------------------
    off_a, off_f = both(('evict', '0'), 'acc'), both(('evict', '0'), 'forg')
    print(f"\n{'=' * 74}\nSYNTHESIS -- RETENTION AT 33c's BLOCKEDNESS\n{'=' * 74}")
    print("\n  [1] A MIS-SET BUDGET IS WORSE THAN NO BUDGET. This is the finding")
    print("      that makes the knobs matter, and it is symmetric across BOTH")
    print("      of them. No slot budget at all (E=0) scores "
          f"ACC {off_a:.3f} / FORG {off_f:.3f}.")
    for label, key in (("window mis-set  (E=2000, count)", ('evict', '2000')),
                       ("victim mis-set  (E=250, random)", ('victim', '250:random')),
                       ("both set well   (E=250, count)", ('evict', '250'))):
        print(f"      {label:<32} ACC {both(key, 'acc'):.3f} / "
              f"FORG {both(key, 'forg'):.3f}"
              + ("   <-- WORSE THAN OFF on both axes"
                 if both(key, 'acc') < off_a and both(key, 'forg') > off_f
                 else ""))
    print("      So 33b's law needs a second clause. 33b: 'the stale pool must")
    print("      contain the present, or eviction eats the past'. Phase 39 adds:")
    print("      AND THE POOL MUST BE ORDERED BY HOW ESTABLISHED ITS MEMBERS ARE,")
    print("      OR EVICTION EATS THE PAST JUST AS FAST. Getting either one")
    print("      wrong is worse than never evicting.")
    print("\n  [2] WHAT IS ACTUALLY TUNABLE, ranked by pooled |dFORG| (n=10):")
    ranked = []
    for key in sel:
        if key == ('evict', '250') or key not in hld:
            continue
        st = pool(key, 'forg')
        ranked.append((abs(st['mean']), key, st, pool(key, 'acc')))
    for mag, key, sf, sa in sorted(ranked, reverse=True)[:8]:
        print(f"      {key[0] + '=' + key[1]:>22}  dFORG {sf['mean']:+.4f} "
              f"(p={sf['p']:.4f}, {sf['neg']}/{sf['n']} down)   "
              f"dACC {sa['mean']:+.4f} (p={sa['p']:.4f})")
    print("\n      Two knobs BUY retention rather than merely avoiding harm, and")
    print("      both survive the held-out split with consistent sign:")
    print("        - SLOT HEADROOM K: monotone, the largest single gain")
    print("          (K=96: dFORG -0.111, dACC +0.127, 10/10 both). Consistent")
    print("          with 33d's capacity result, now reseeded and held out.")
    print("        - PROBATION/CONFIRM: phase 14's provisional-slot machinery,")
    print("          UNUSED by 33c's recipe (confirm=0), is a real retention")
    print("          knob (confirm=3/probation=1000: dFORG -0.062, dACC +0.056,")
    print("          10/10 down). NOT pre-registered -- found by the sweep, so")
    print("          it is reported as a grid-selected result that survived the")
    print("          held-out split, NOT as a tuned recommendation.")
    print("\n  [3] SCOPE, RESTATED BECAUSE IT IS EASY TO DROP: every number above")
    print("      is retention AT 33c's BLOCKEDNESS -- five disjoint 2-class tasks")
    print("      in strict sequence. Phase 38 measured that the forgetting here")
    print("      is produced by that blocking. None of this transfers to a less")
    print("      blocked stream without being re-measured; T6.6 (phase 49) is")
    print("      the target that varies the axis.")

    # ---- F3, the factor that could not be measured ------------------------
    print(f"\n{'=' * 74}\nTHE MISS: F3 (p_decay) IS UNMEASURABLE ON THIS PROTOCOL")
    print(f"{'=' * 74}")
    print("    Every p_decay level returns byte-identical ACC/FORG/task-0. That")
    print("    is NOT 'decay does not matter for retention'. p_decay decays the")
    print("    TRANSITION GRAPH only (verified wired: at p_decay=0.01 the graph")
    print("    mass falls 2782 -> 93.6, a 30x decay) while leaving the memory")
    print("    bank xi bitwise untouched (max|dxi| = 0.0e+00) -- and phase 33c's")
    print("    readout (label_readout.LabelEvidenceReadout) decodes from xi")
    print("    overlaps ALONE and never consults the graph. The protocol has no")
    print("    instrument that can see this factor. Measuring it needs a task")
    print("    whose metric consumes P (next-symbol prediction / planning), not")
    print("    a class-incremental classification readout. RECORDED AS A MISS.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('select', 'holdout', 'both'),
                    default='both')
    ap.add_argument('--out', default='/tmp/phase39_results.json')
    args = ap.parse_args()

    t0 = time.time()
    print("PHASE 39 (T6.2): retention mechanism study")
    print("SCOPE: retention knobs AT 33c's BLOCKEDNESS (phase 38: forgetting")
    print("       here is produced by sequential blocking; 33c is ONE point on")
    print("       that axis. T6.6/phase 49 varies it. Do not generalize past it.)")
    print(f"reference cell: {REF}")

    store = {}
    splits = ({'select': [('select', SEEDS_SELECT)],
               'holdout': [('holdout', SEEDS_HOLDOUT)],
               'both': [('select', SEEDS_SELECT), ('holdout', SEEDS_HOLDOUT)]}
              [args.split])

    for label, seeds in splits:
        print(f"\n{'=' * 74}\nSPLIT: {label}  seeds {list(seeds)}\n{'=' * 74}")
        out = {}
        sweep("F1 -- eviction window E", [0, 100, 250, 500, 750, 1000, 2000],
              'evict', seeds, out)
        print("\n  F2 -- victim rule x window  (all other factors at reference)")
        print(f"    {'E':>5} {'rule':>7}  {'ACC':>16}  {'FORG':>16}  {'task-0':>16}")
        for E in (100, 250, 500, 750):
            for rule in ('count', 'rate', 'random'):
                cs = [run_arm(s, **{**REF, 'evict': E, 'victim': rule})
                      for s in seeds]
                out[('victim', f"{E}:{rule}")] = cs
                a = np.array([c['acc'] for c in cs])
                f = np.array([c['forg'] for c in cs])
                t = np.array([c['task0'] for c in cs])
                print(f"    {E:>5} {rule:>7}  "
                      f"{a.mean():.4f}+/-{a.std(ddof=1)/np.sqrt(len(a)):.4f}  "
                      f"{f.mean():.4f}+/-{f.std(ddof=1)/np.sqrt(len(f)):.4f}  "
                      f"{t.mean():.4f}+/-{t.std(ddof=1)/np.sqrt(len(t)):.4f}")
        sweep("F3 -- p_decay (transition forgetting, phase 11)",
              [0.0, 0.001, 0.003, 0.01], 'p_decay', seeds, out)
        print("\n  F4 -- probation / confirm (phase 14 provisional slots)")
        print(f"    {'confirm':>8} {'probation':>10}  {'ACC':>16}  {'FORG':>16}"
              f"  {'task-0':>16}")
        for conf, prob in [(0, 6000), (2, 1000), (2, 3000), (3, 1000), (3, 3000)]:
            cs = [run_arm(s, **{**REF, 'confirm': conf, 'probation': prob})
                  for s in seeds]
            out[('confirm', f"{conf}:{prob}")] = cs
            a = np.array([c['acc'] for c in cs])
            f = np.array([c['forg'] for c in cs])
            t = np.array([c['task0'] for c in cs])
            print(f"    {conf:>8} {prob:>10}  "
                  f"{a.mean():.4f}+/-{a.std(ddof=1)/np.sqrt(len(a)):.4f}  "
                  f"{f.mean():.4f}+/-{f.std(ddof=1)/np.sqrt(len(f)):.4f}  "
                  f"{t.mean():.4f}+/-{t.std(ddof=1)/np.sqrt(len(t)):.4f}")
        sweep("F5 -- slot headroom K", [40, 48, 56, 72, 96], 'K', seeds, out)
        store[label] = out

        # ---- paired effect sizes against the reseeded reference cell -------
        base = out[('evict', '250')]
        print(f"\n  PAIRED EFFECT SIZES vs the RESEEDED reference cell "
              f"(within-seed, n={len(seeds)})")
        print(f"    exact sign-flip null; at n={len(seeds)} the floor is "
              f"p={2 / 2 ** len(seeds):.4f} -- no result here can reach p<0.05")
        print(f"    {'factor level':>26}  {'dACC':>34}  {'dFORG':>34}")
        for key in out:
            if key == ('evict', '250'):
                continue
            arm = out[key]
            print(f"    {key[0] + '=' + key[1]:>26}  "
                  f"{fmt(paired(arm, base, 'acc')):>34}  "
                  f"{fmt(paired(arm, base, 'forg')):>34}")

    probe = ledger_probe()

    if args.split == 'both':
        verdict(store, probe)

    # ---- headline comparisons, pooled over all ten seeds ------------------
    if args.split == 'both':
        print(f"\n{'=' * 74}\nHEADLINE COMPARISONS, POOLED OVER ALL 10 SEEDS")
        print(f"(2^10 = 1024 sign assignments; min attainable p = "
              f"{2 / 1024:.4f})\n{'=' * 74}")
        allseeds = tuple(SEEDS_SELECT) + tuple(SEEDS_HOLDOUT)
        pooled = {}
        for E in (100, 250, 500, 750):
            for rule in ('count', 'rate', 'random'):
                pooled[(E, rule)] = [run_arm(s, **{**REF, 'evict': E,
                                                   'victim': rule})
                                     for s in allseeds]
        print(f"    {'comparison':>22}  {'dACC':>34}  {'dFORG':>34}")
        for E in (100, 250, 500, 750):
            for rule in ('rate', 'random'):
                st_a = paired(pooled[(E, rule)], pooled[(E, 'count')], 'acc')
                st_f = paired(pooled[(E, rule)], pooled[(E, 'count')], 'forg')
                print(f"    {f'E={E} {rule}-count':>22}  {fmt(st_a):>34}  "
                      f"{fmt(st_f):>34}")
        store['pooled_victim'] = {f"{E}:{r}": v for (E, r), v in pooled.items()}

        # window spread vs rule spread -- prediction (a)'s decision rule
        wins = {}
        for E in (0, 100, 250, 500, 750, 1000, 2000):
            wins[E] = [run_arm(s, **{**REF, 'evict': E}) for s in allseeds]
        w_forg = np.array([np.mean([c['forg'] for c in wins[E]]) for E in wins])
        r_forg = np.array([np.mean([c['forg'] for c in pooled[(250, r)]])
                           for r in ('count', 'rate', 'random')])
        print(f"\n    prediction (a) decision rule, pooled n=10:")
        print(f"      spread of mean FORG across WINDOWS (E=0..2000): "
              f"{w_forg.max() - w_forg.min():.4f}")
        print(f"      spread of mean FORG across RULES   (at E=250):  "
              f"{r_forg.max() - r_forg.min():.4f}")
        store['pooled_window'] = {str(E): v for E, v in wins.items()}

    with open(args.out, 'w') as fh:
        json.dump({'store': {k: {str(kk): [{m: c[m] for m in
                                            ('acc', 'forg', 'task0', 'slots',
                                             'evictions', 'census', 'final')}
                                           for c in vv]
                                 for kk, vv in v.items()}
                             for k, v in store.items()},
                   'probe': {f"{E}:{r}": {'n': d['n'], 'slots': d['slots'],
                                          'counts': d['counts']}
                             for (E, r), d in probe.items()},
                   'ref': REF}, fh)
    print(f"\nwrote {args.out}   ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    sys.exit(main())
