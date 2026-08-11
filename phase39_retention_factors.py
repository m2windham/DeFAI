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
RESULTS
===========================================================================
Filled in by the committed run; see ROADMAP row 39 for the narrative version.
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
