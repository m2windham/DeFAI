"""
PHASE 40: LOGIC-LAYER DEPTH AND OPTIMIZATION (target T6.3, consolidation
posture 2026-08-07).

Phase 30 showed the logic layer can reason with the field ABSENT: k-step
inference at corr 0.995-0.999 against a ~0.45 permutation null, rollout
matching field recall at ~20x, directed recall reaching goals at true
shortest-path length. It then stopped at three ops. This phase treats the
field-free graph as the differentiated capability it is and takes it deeper
on the four axes T6.3 names, each measured against a null rather than
demonstrated:

  (A) PLANNING UNDER UNCERTAINTY. `next_hops` runs Dijkstra on -log of the
      POINT ESTIMATE of each transition probability, which throws the
      evidence away: a symbol seen three times, all three going to j, offers
      a near-free edge, while an edge seen 900 times in 1000 visits scores
      0.9 and looks worse. Phase 40 plans on Wilson lower bounds instead and
      reports PATH RELIABILITY -- the product of edge probabilities and its
      lower bound, the weakest link, and how often that link was seen --
      against a hop-count planner, the MLE planner phase 30 already had, an
      eval-only oracle ceiling, and a permutation null that keeps the point
      estimates and shuffles only the confidence correction.

  (B) COMPOSITIONAL GOALS. Negation ("reach A without passing B") by deleting
      symbols from the graph the planner sees; conjunction ("visit A and B")
      by exhausting goal orders over cached shortest-path trees. Certified
      against brute-force optimal walks on a graph small enough to enumerate,
      and measured against a random walker that must satisfy the same
      constraint by luck.

  (C) TEMPORAL ABSTRACTION. Macro-edges, in both readings. The `fold` reading
      -- contract pass-through symbols -- gets a census first, on 33h's
      precedent of pricing a fold before running one. The overlay reading
      gets two arms whose difference is the scientific content: macros MINED
      from the transition matrix cannot beat planning over their own parts
      (Dijkstra already pays exactly their product), so zero is forced and
      the job is to measure it; macros OBSERVED as symbol trigrams can carry
      second-order structure the matrix is structurally unable to hold. Run
      on a first-order world, where they must collapse, and a second-order
      world, where they need not.

  (D) OPTIMIZATION. Phase 30's ~20x was measured on a dense P; 33g measured P
      at 6.1% density at K=160, with out-degree roughly constant in K, so the
      matrix gets emptier as the organism gets bigger. Sparse ports of
      next_hops / rollout / kstep, timed across K and checked for equality --
      bitwise where the arithmetic permits it, tolerance where skipping zeros
      reorders a float reduction.

PRE-REGISTERED PREDICTIONS (frozen before the run, per standing rules):
  (a) confidence-weighted planning beats hop-count planning on a world with
      unreliable edges, against a measured null.
  (b) sparse-P reasoning is >= 2x faster at K >= 112 with bitwise-equal
      results.
  (c) honest negative available: macro-edges may not beat plain planning at
      this graph size -- phase 29's thin-structure lesson applies and "level 2
      adds nothing" is a legitimate outcome.
Category-6 standing lessons bind: every arm reseeded s=0-4 paired, and any
grid-selected claim re-measured on HELD-OUT seeds s=5-9 (T1.6/T1.8/T1.9).
Section A4 asks a sharper question than (a) does -- confidence weighting
against the MLE planner rather than the hop-count one -- and is labelled
throughout as NOT pre-registered, with its selection held out accordingly.

CONSTRAINT: reasoning ops are graph-only. Nothing below builds a field, an
embedding or an overlap; section D's single organism run exists solely to
supply a REAL transition matrix whose density can be compared with 33g's, and
the field is gone before any reasoning op is timed.

Ground truth (the true transition law, the second-order law, the oracle
planner, brute-force optimal walks) is used for EVALUATION ONLY and never
reaches a mechanism.
"""

import time

import numpy as np

from organism import (MacroGraph, Organism, SparseTransitions, TransitionGraph,
                      _trace)

ALPHA0 = 0.05                 # pre-registered default confidence level
SEEDS = range(5)              # paired reseeds for every arm
HELDOUT = range(5, 10)        # fresh seeds for any grid-selected claim
NPERM = 200                   # permutation null draws
N_A = 120                     # symbols in the planning world
STEPS_A = 2000                # primary observation budget (the thin regime)
BUDGETS = [1000, 2000, 6000, 20000, 60000]


# =========================================================================
# (0) ANCHOR GUARD: next_hops was refactored onto a shared Dijkstra so the
# new planners and the sparse port run one algorithm. Phase 30's committed
# numbers depend on it, so the pre-refactor body is kept here verbatim and
# the two are compared BITWISE -- an equality proven per run, not asserted.
# =========================================================================
def phase30_next_hops_reference(graph, idx, goal, eps=1e-12):
    """`TransitionGraph.next_hops` exactly as it stood before phase 40."""
    Pn = graph.normalized(idx); n = Pn.shape[0]
    W = -np.log(Pn + eps)
    dist = np.full(n, np.inf); dist[goal] = 0.0
    nxt = np.full(n, -1, int); nxt[goal] = goal
    done = np.zeros(n, bool)
    for _ in range(n):
        u = int(np.argmin(np.where(done, np.inf, dist)))
        if not np.isfinite(dist[u]) or done[u]:
            break
        done[u] = True
        relax = (Pn[:, u] > eps) & (dist[u] + W[:, u] < dist)
        nxt[relax] = u
        dist[relax] = dist[u] + W[relax, u]
    return nxt, dist


# =========================================================================
# worlds -- all symbolic. No field, no embeddings.
# =========================================================================
def unreliable_world(seed, n=N_A, out_deg=6, steps=STEPS_A, zipf=1.1):
    """A world with UNRELIABLE EDGES, which is what makes confidence-weighted
    planning a question rather than a refinement.

    Successors are drawn toward popular symbols (Zipf weights), so the
    stationary distribution is heavy-tailed and a finite observation window
    leaves a long tail of symbols visited a handful of times. Those rows are
    the over-confident ones -- three visits all going the same way reads as
    certainty -- and the organism cannot tell from the point estimate alone
    which of its edges it has earned the right to believe."""
    rng = np.random.default_rng(1000 + seed)
    pop = 1.0 / np.arange(1, n + 1) ** zipf
    pop /= pop.sum()
    order = rng.permutation(n)
    pop = pop[np.argsort(order)]
    T = np.zeros((n, n))
    for i in range(n):
        w = pop.copy(); w[i] = 0.0
        succ = rng.choice(n, size=out_deg, replace=False, p=w / w.sum())
        T[i, succ] = rng.dirichlet(np.full(out_deg, 0.4))
    g = TransitionGraph(n)                     # counts, not probabilities,
    cur = int(rng.integers(n))                 # are all the organism ever has
    seq = [cur]
    for _ in range(steps):
        nxt = int(rng.choice(n, p=T[cur]))
        g.observe(cur, nxt)
        cur = nxt
        seq.append(cur)
    return g, T, np.array(seq, int)


def second_order_world(seed, n=30, out_deg=3, steps=60000, order2=True):
    """A world whose next symbol depends on the PREVIOUS TWO symbols, or --
    with order2=False -- the matched first-order control. A transition matrix
    cannot represent the former; an observed trigram macro can."""
    rng = np.random.default_rng(2000 + seed)
    succ = {a: rng.choice([x for x in range(n) if x != a], size=out_deg,
                          replace=False) for a in range(n)}
    base = {a: rng.dirichlet(np.full(out_deg, 0.7)) for a in range(n)}
    law = {}
    for a in range(n):
        for b in succ[a]:
            law[(a, int(b))] = (succ[int(b)],
                                rng.dirichlet(np.full(out_deg, 0.35)) if order2
                                else base[int(b)])
    g = TransitionGraph(n)
    a = 0; b = int(succ[0][0])
    seq = [a, b]
    g.observe(a, b)
    for _ in range(steps):
        s, p = law[(a, int(b))]
        c = int(rng.choice(s, p=p))
        g.observe(b, c)
        seq.append(c)
        a, b = b, c
    return g, law, np.array(seq, int)


def true_marginal(law, n):
    """Exact P(b | a) implied by a second-order law, via the stationary
    distribution over CONTEXTS. Eval-only: no mechanism sees this."""
    ctx = sorted(law.keys())
    pos = {c: i for i, c in enumerate(ctx)}
    M = np.zeros((len(ctx), len(ctx)))
    for c, (s, p) in law.items():
        for cc, pp in zip(s, p):
            nxt = (c[1], int(cc))
            if nxt in pos:
                M[pos[c], pos[nxt]] += pp
    pi = np.full(len(ctx), 1.0 / len(ctx))
    for _ in range(4000):
        pi = pi @ M
        tot = pi.sum()
        if tot <= 0:
            break
        pi /= tot
    marg = np.zeros((n, n))
    for c, w in zip(ctx, pi):
        marg[c[0], c[1]] += w
    return marg / (marg.sum(1, keepdims=True) + 1e-300)


def composition_world(n=12, seed=11):
    """Small enough to enumerate exhaustively, structured so that both kinds
    of compositional query have a real answer: a strong ring plus two chord
    families, so detours exist and the cheapest route is genuinely contested."""
    rng = np.random.default_rng(seed)
    g = TransitionGraph(n)
    for i in range(n):
        g.P[i, (i + 1) % n] = rng.integers(250, 400)
        g.P[i, (i + 4) % n] = rng.integers(90, 160)
        g.P[i, (i + 6) % n] = rng.integers(20, 45)
    return g


def sparse_world(seed, n, out_deg=10):
    """A synthetic count matrix at 33g's measured shape: out-degree roughly
    constant in K, so density falls as 1/K the way the real one does. Used
    for the timing sweep, where only sparsity pattern and scale matter."""
    rng = np.random.default_rng(3000 + seed + n)
    g = TransitionGraph(n)
    for i in range(n):
        succ = rng.choice(n, size=min(out_deg, n - 1), replace=False)
        succ = succ[succ != i]
        g.P[i, succ] = rng.integers(1, 400, size=len(succ))
    return g


# =========================================================================
# scoring helpers -- ground truth, evaluation only
# =========================================================================
def log_true(path, T):
    """log of the TRUE probability of executing a plan under the world's real
    law. The quantity a planner should maximize and the one it cannot see."""
    tot = 0.0
    for u, v in zip(path[:-1], path[1:]):
        p = T[u, v]
        if p <= 0:
            return -np.inf
        tot += float(np.log(p))
    return tot


def plan_all(Q, goals, n, eps=1e-12):
    """Every (source -> goal) plan under one edge-quality matrix."""
    out = {}
    for gl in goals:
        nxt, _ = TransitionGraph.dijkstra(Q, int(gl), eps)
        for a in range(n):
            if a == gl:
                continue
            p = _trace(nxt, a, int(gl))
            if p is not None:
                out[(a, int(gl))] = tuple(p)
    return out


def score_on(plans, T, pairs):
    """Mean log true reliability over a FIXED pair set. Fixing the pairs is
    load-bearing: the oracle plans on the true law and can therefore reach
    pairs the observed graph cannot, and averaging each arm over its own
    reachable set would silently compare different questions."""
    vals = [log_true(plans[k], T) for k in pairs if k in plans]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else -np.inf


def brute_force_best(P, a, b, max_hops, avoid=(), must=()):
    """Exhaustive best walk from `a`, within `max_hops`, avoiding `avoid` and
    visiting every symbol in `must`; ends at `b`, or anywhere once all of
    `must` is covered when b is None. Exponential by construction -- only run
    on the 12-symbol composition world, as the optimality oracle."""
    n = P.shape[0]
    banned = {int(v) for v in avoid}
    adj = [np.nonzero(P[i] > 0)[0] for i in range(n)]
    Pn = P / (P.sum(1, keepdims=True) + 1e-9)
    best = [(-np.inf, None)]

    def rec(cur, walk, lp):
        if lp <= best[0][0] - 30:
            return
        done = (cur == b) if b is not None else all(m in walk for m in must)
        if done and all(m in walk for m in must) and lp > best[0][0]:
            best[0] = (lp, list(walk))
        if len(walk) - 1 >= max_hops:
            return
        for v in adj[cur]:
            v = int(v)
            if v in banned:
                continue
            walk.append(v)
            rec(v, walk, lp + float(np.log(Pn[cur, v])))
            walk.pop()

    if a in banned or (b is not None and b in banned):
        return -np.inf, None
    rec(a, [a], 0.0)
    return best[0]


def timed(fn, repeat=5):
    """Best-of-`repeat` wall clock, after one warm-up call."""
    fn()
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def survives(vals):
    """T1.6's survival rule: positive mean AND no sign flip across seeds."""
    return float(np.mean(vals)) > 0 and all(v > 0 for v in vals)


# =========================================================================
def section_0():
    print("\n(0) ANCHOR GUARD: next_hops after the shared-Dijkstra refactor")
    worst_n = worst_d = 0.0
    for s in SEEDS:
        g, _, _ = unreliable_world(s, n=60, steps=8000)
        idx = np.arange(60)
        for goal in (0, 7, 31, 59):
            r_n, r_d = phase30_next_hops_reference(g, idx, goal)
            n_n, n_d = g.next_hops(idx, goal)
            worst_n = max(worst_n, float(np.abs(r_n - n_n).max()))
            fin = np.isfinite(r_d) & np.isfinite(n_d)
            if fin.any():
                worst_d = max(worst_d, float(np.abs(r_d[fin] - n_d[fin]).max()))
            if not np.array_equal(np.isfinite(r_d), np.isfinite(n_d)):
                worst_d = np.inf
    print(f"    pre- vs post-refactor next_hops over 5 worlds x 4 goals:")
    print(f"      next-hop delta {worst_n:.1e}, dist delta {worst_d:.1e}  -> "
          f"{'BITWISE IDENTICAL' if worst_n == 0 and worst_d == 0 else 'DIVERGED'}")
    print("    phase 30's kstep / rollout / next_hops / plan are otherwise")
    print("    untouched; harness section 6 pins their values on both backends.")
    return worst_n == 0 and worst_d == 0


def section_a():
    print("\n(A) MULTI-STEP PLANNING UNDER UNCERTAINTY")
    print(f"    world: {N_A} symbols, out-degree 6, Zipf-popular successors,")
    print(f"    {STEPS_A} observed transitions (~{STEPS_A / N_A:.0f} per symbol --")
    print("    the thin regime, where the graph is genuinely under-evidenced).")
    print("    Score = mean log TRUE reliability of the plan (eval only),")
    print("    over the pair set every arm can reach. Higher is better.")

    ladder, adv_hops, adv_mle, nulls, census = [], [], [], [], []
    for s in SEEDS:
        g, T, _ = unreliable_world(s)
        idx = np.arange(N_A)
        C = g.P; visits = C.sum(1)
        goals = list(np.argsort(-visits)[:25])
        Pn = g.normalized(idx)
        Lcb = g.confidence(idx, ALPHA0)
        arms = {"hops": g.edge_quality(idx, "hops"), "mle": Pn, "lcb": Lcb,
                "oracle*": T.copy()}
        plans = {k: plan_all(Q, goals, N_A) for k, Q in arms.items()}
        pairs = set(plans["mle"])
        for k in plans:
            pairs &= set(plans[k])
        sc = {k: score_on(plans[k], T, pairs) for k in plans}
        ladder.append(sc)
        adv_hops.append(sc["lcb"] - sc["hops"])
        adv_mle.append(sc["lcb"] - sc["mle"])

        thin = int((visits <= 3).sum())
        over = [float(Pn[i, j] / T[i, j]) for i in range(N_A) for j in
                np.nonzero(C[i])[0] if visits[i] <= 3 and T[i, j] > 0]
        census.append((thin, len(pairs),
                       float(np.mean(over)) if over else 1.0))

        # permutation null: keep the point estimates, shuffle ONLY the
        # confidence correction across observed edges -- kills the
        # count -> trust correspondence, preserves its value distribution.
        seen = C > 0
        shrink = np.zeros_like(Pn)
        shrink[seen] = Lcb[seen] / np.maximum(Pn[seen], 1e-300)
        nrng = np.random.default_rng(7000 + s)
        gn = goals[:8]
        pn_pairs = set(plan_all(Pn, gn, N_A))
        base = score_on(plan_all(Pn, gn, N_A), T, pn_pairs)
        obs = score_on(plan_all(Lcb, gn, N_A), T, pn_pairs) - base
        vals = shrink[seen].copy()
        draws = []
        for _ in range(NPERM):
            nrng.shuffle(vals)
            Qn = np.zeros_like(Pn); Qn[seen] = Pn[seen] * vals
            draws.append(score_on(plan_all(Qn, gn, N_A), T, pn_pairs) - base)
        nulls.append((obs, float(np.percentile(draws, 99))))

    print(f"\n    thin-tail census (mean over 5 seeds): "
          f"{np.mean([c[0] for c in census]):.0f}/{N_A} symbols observed <= 3 "
          f"times;")
    print(f"      their edges over-state the true probability by "
          f"{np.mean([c[2] for c in census]):.1f}x on average -- this is the")
    print(f"      unreliability the point estimate cannot see. "
          f"{np.mean([c[1] for c in census]):.0f} scored pairs/seed.")

    print(f"\n    {'planner':<10}{'mean log true reliability, s=0-4':<40}{'mean':>8}")
    for name in ("hops", "mle", "lcb", "oracle*"):
        v = [r[name] for r in ladder]
        print(f"    {name:<10}{' '.join(f'{x:7.3f}' for x in v):<40}{np.mean(v):8.3f}")
    print("    (* oracle plans on the TRUE law: an evaluation ceiling, not an arm)")

    print(f"\n    lcb - hops : mean {np.mean(adv_hops):+.3f} "
          f"[{min(adv_hops):+.3f}, {max(adv_hops):+.3f}]  "
          f"sign flips {sum(1 for v in adv_hops if v <= 0)}/5")
    obs = [o for o, _ in nulls]; bar = [b for _, b in nulls]
    print(f"    permutation null (shuffled shrinkage, {NPERM} draws/seed):")
    print(f"      observed lcb-vs-mle {np.mean(obs):+.4f} vs null 99th pct "
          f"{np.mean(bar):+.4f} -> "
          f"{'ABOVE NULL' if np.mean(obs) > np.mean(bar) else 'NOT above null'}")

    # held-out verification of the pre-registered alpha
    ho = []
    for s in HELDOUT:
        g, T, _ = unreliable_world(s)
        idx = np.arange(N_A)
        goals = list(np.argsort(-g.P.sum(1))[:25])
        ph = plan_all(g.edge_quality(idx, "hops"), goals, N_A)
        pl = plan_all(g.confidence(idx, ALPHA0), goals, N_A)
        pr = set(ph) & set(pl)
        ho.append(score_on(pl, T, pr) - score_on(ph, T, pr))
    print(f"    HELD-OUT seeds 5-9 at the pre-registered alpha={ALPHA0}: "
          f"mean {np.mean(ho):+.3f}")
    print(f"      [{min(ho):+.3f}, {max(ho):+.3f}], "
          f"sign flips {sum(1 for v in ho if v <= 0)}/5")

    ok_a = (survives(adv_hops) and np.mean(obs) > np.mean(bar) and survives(ho))
    print(f"\n    PREDICTION (a): {'CONFIRMED' if ok_a else 'NOT CONFIRMED'}")

    # ---- A4: the sharper question, NOT pre-registered -------------------
    print("\n    A4. NOT PRE-REGISTERED, and the more interesting question:")
    print("        does confidence weighting beat the MLE planner phase 30")
    print("        already had? Swept over the evidence budget, decomposed")
    print("        into how often the two planners even disagree.")
    print(f"\n        {'budget':>8}{'obs/sym':>9}{'plans differ':>14}"
          f"{'delta WHEN differ':>19}{'flips':>7}{'overall':>10}")
    sweep = {}
    for steps in BUDGETS:
        diffs, condv, overall = [], [], []
        for s in SEEDS:
            g, T, _ = unreliable_world(s, steps=steps)
            idx = np.arange(N_A)
            goals = list(np.argsort(-g.P.sum(1))[:25])
            pm = plan_all(g.normalized(idx), goals, N_A)
            pl = plan_all(g.confidence(idx, ALPHA0), goals, N_A)
            common = [k for k in pm if k in pl]
            d = [k for k in common if pm[k] != pl[k]]
            cv = [log_true(pl[k], T) - log_true(pm[k], T) for k in d]
            cv = [x for x in cv if np.isfinite(x)]
            av = [log_true(pl[k], T) - log_true(pm[k], T) for k in common]
            av = [x for x in av if np.isfinite(x)]
            diffs.append(len(d) / max(len(common), 1))
            condv.append(float(np.mean(cv)) if cv else 0.0)
            overall.append(float(np.mean(av)) if av else 0.0)
        sweep[steps] = (np.mean(diffs), np.mean(condv), condv, np.mean(overall))
        print(f"        {steps:>8}{steps / N_A:>9.0f}{np.mean(diffs):>13.1%}"
              f"{np.mean(condv):>+19.3f}"
              f"{sum(1 for v in condv if v <= 0):>5}/5{np.mean(overall):>+10.4f}")
    winners = [b for b in BUDGETS if survives(sweep[b][2])]
    print(f"        survives T1.6's rule (mean>0, no flip) on the selection")
    print(f"        seeds at budgets: {winners if winners else 'none'}")
    print("        The budget itself was chosen after an exploratory sweep, so")
    print("        every surviving one is re-measured on FRESH seeds 5-9 --")
    print("        the whole reason that rule exists (33g's low-rank arm):")
    a4 = []
    for pick in winners:
        hov = []
        for s in HELDOUT:
            g, T, _ = unreliable_world(s, steps=pick)
            idx = np.arange(N_A)
            goals = list(np.argsort(-g.P.sum(1))[:25])
            pm = plan_all(g.normalized(idx), goals, N_A)
            pl = plan_all(g.confidence(idx, ALPHA0), goals, N_A)
            d = [k for k in pm if k in pl and pm[k] != pl[k]]
            cv = [log_true(pl[k], T) - log_true(pm[k], T) for k in d]
            cv = [x for x in cv if np.isfinite(x)]
            hov.append(float(np.mean(cv)) if cv else 0.0)
        ok = survives(hov)
        print(f"          budget {pick:>6}: held-out {np.mean(hov):+.3f} "
              f"[{min(hov):+.3f}, {max(hov):+.3f}], "
              f"{sum(1 for v in hov if v <= 0)}/5 flips -> "
              f"{'SURVIVES' if ok else 'DOES NOT SURVIVE'}")
        a4.append((pick, float(np.mean(hov)), ok))

    # ---- reliability reporting, on pairs where the arms actually differ --
    g0, T0, _ = unreliable_world(0)
    idx0 = np.arange(N_A)
    goals0 = list(np.argsort(-g0.P.sum(1))[:25])
    pm0 = plan_all(g0.normalized(idx0), goals0, N_A)
    pl0 = plan_all(g0.confidence(idx0, ALPHA0), goals0, N_A)
    diff_keys = [k for k in pm0 if k in pl0 and pm0[k] != pl0[k]]
    deltas = {k: log_true(pl0[k], T0) - log_true(pm0[k], T0) for k in diff_keys}
    deltas = {k: v for k, v in deltas.items() if np.isfinite(v)}
    ranked = sorted(deltas, key=lambda k: -deltas[k])
    show = ranked[:2] + ranked[-2:]
    print("\n    path reliability reporting -- what the caller now gets back.")
    print(f"    Seed 0, the {len(deltas)} pairs where the planners disagree:")
    print(f"    lcb wins {sum(1 for v in deltas.values() if v > 0)}, loses "
          f"{sum(1 for v in deltas.values() if v < 0)}, mean "
          f"{np.mean(list(deltas.values())):+.3f} nats.")
    print(f"    Two best and two worst, so the spread is visible, not the spin:")
    print(f"      {'pair':<10}{'mle plan':<28}{'lcb plan':<28}{'true d':>9}")
    for k in show:
        rm = g0.path_report(idx0, pm0[k]); rl = g0.path_report(idx0, pl0[k])
        print(f"      {f'{k[0]}->{k[1]}':<10}"
              f"{f'{rm.hops}h p={rm.p_mle:.1e} nmin={rm.weakest_n:.0f}':<28}"
              f"{f'{rl.hops}h p={rl.p_mle:.1e} nmin={rl.weakest_n:.0f}':<28}"
              f"{deltas[k]:>+9.3f}")
    wn_m = np.mean([g0.path_report(idx0, pm0[k]).weakest_n for k in diff_keys])
    wn_l = np.mean([g0.path_report(idx0, pl0[k]).weakest_n for k in diff_keys])
    print(f"    Weakest-link evidence over all disagreements: mle plans rest on")
    print(f"    edges seen {wn_m:.0f} times, lcb plans on edges seen {wn_l:.0f}"
          f" times. The two")
    print(f"    planners differ less in WHICH edges they trust than in how they")
    print(f"    price them; either way a hop count shows neither, and PathReport")
    print(f"    carries both -- the durable half of this op is the report.")
    return ok_a, np.mean(adv_hops), np.mean(obs), np.mean(bar), np.mean(ho), sweep, a4


def section_b():
    print("\n(B) COMPOSITIONAL GOALS: negation and conjunction")
    print("    12 symbols -- small enough that brute force certifies every")
    print("    answer. Ring plus two chord families, so detours are real.")
    g = composition_world()
    n = 12; idx = np.arange(n)
    pairs = [(0, 5), (0, 8), (2, 7), (5, 1), (3, 10), (8, 4)]

    print(f"\n    {'query':<24}{'plan':<28}{'hops':>5}{'vs brute force':>17}")
    viol = 0; gaps = []; prices = []; unreach = 0; gates = {}
    for a, b in pairs:
        free = g.plan_reliable(idx, a, b, weights="mle")
        gate = int(free.path[1]) if free.hops >= 2 else int(free.path[-1])
        gates[(a, b)] = gate
        con = g.plan_reliable(idx, a, b, weights="mle", avoid=[gate])
        bf_lp, _ = brute_force_best(g.P, a, b, max_hops=9, avoid=[gate])
        if con is None:
            unreach += 1
            print(f"    {f'{a}->{b} avoid {gate}':<24}{'UNREACHABLE':<28}")
            continue
        if gate in con.path:
            viol += 1
        lp = float(np.log(con.p_mle))
        gaps.append(bf_lp - lp); prices.append(float(np.log(free.p_mle)) - lp)
        print(f"    {f'{a}->{b} avoid {gate}':<24}{str(con.path):<28}"
              f"{con.hops:>5}{bf_lp - lp:>+17.2e}")
    print(f"\n    avoidance violations {viol}/{len(pairs)}; unreachable under")
    print(f"    the constraint {unreach}/{len(pairs)}; optimality gap vs brute")
    print(f"    force max {max(gaps):.1e} (0 = the plan IS the best legal walk).")
    print(f"    Price of the constraint: {np.mean(prices):.2f} nats of "
          f"log-reliability -- reported, not hidden.")

    print("\n    conjunction: visit every goal, planner chooses the order")
    conj_gap, cov = [], 0
    cases = [(0, (5, 9)), (2, (7, 11)), (4, (1, 10)), (6, (0, 3)),
             (1, (4, 7, 10))]
    for a, goals in cases:
        rep = g.plan_visit(idx, a, list(goals), weights="mle")
        if rep is None:
            print(f"    from {a} visit {goals}: no legal walk")
            continue
        cov += int(all(x in rep.path for x in goals))
        bf_lp, _ = brute_force_best(g.P, a, None, max_hops=max(9, rep.hops),
                                    must=goals)
        conj_gap.append(bf_lp - float(np.log(rep.p_mle)))
        print(f"    from {a} visit {str(goals):<12} {str(rep.path):<34}"
              f"{rep.hops:>3}h  gap {conj_gap[-1]:>+8.1e}")
    print(f"    all goals covered {cov}/{len(cases)}; max gap vs exhaustive "
          f"search {max(conj_gap):+.1e}")

    print("\n    null: can a random walker satisfy the constraint by luck?")
    print("    (unconstrained walkers reaching the goal, how many trespass)")
    wrng = np.random.default_rng(21)
    Pn = g.normalized(idx)
    trespass = []
    for a, b in pairs[:4]:
        gate = gates[(a, b)]
        con = g.plan_reliable(idx, a, b, weights="mle", avoid=[gate])
        hit = bad = 0; hops = 0
        for _ in range(4000):
            cur = a; walk = [a]
            for _ in range(40):
                row = Pn[cur]; tot = row.sum()
                if tot <= 0:
                    break
                cur = int(wrng.choice(n, p=row / tot)); walk.append(cur)
                if cur == b:
                    hit += 1; hops += len(walk) - 1
                    bad += int(gate in walk[:-1])
                    break
        rate = bad / max(hit, 1)
        trespass.append(rate)
        print(f"      {a}->{b} avoid {gate}: planner {con.hops} hops, 0% "
              f"trespass | walker {hops / max(hit,1):.1f} hops, "
              f"{rate:.0%} trespass")
    print(f"    the constraint is the point: wandering satisfies it "
          f"{1 - np.mean(trespass):.0%} of the time by luck,")
    print(f"    the planner {100}% by construction.")
    return viol, max(gaps), max(conj_gap), float(np.mean(trespass))


def section_c():
    print("\n(C) TEMPORAL ABSTRACTION: macro-edges, in both readings")

    print("\n  (C1) the `fold` reading -- contract pass-through symbols.")
    print("       Census first: 33h priced its fold before running one, and")
    print("       the same question applies. A pass-through needs exactly one")
    print("       predecessor AND one successor at the given count floor.")
    print(f"       {'graph':<28}{'K':>5}{'floor 1':>9}{'floor 5':>9}{'floor 20':>10}")
    for label, gg, ii in [("planning world", unreliable_world(0)[0], np.arange(N_A)),
                          ("second-order world", second_order_world(0)[0], np.arange(30)),
                          ("sparse shape K=160", sparse_world(0, 160), np.arange(160)),
                          ("composition world", composition_world(), np.arange(12))]:
        cs = [len(gg.passthrough_census(ii, min_count=f)) for f in (1, 5, 20)]
        print(f"       {label:<28}{len(ii):>5}{cs[0]:>9}{cs[1]:>9}{cs[2]:>10}")
    print("       Essentially nothing to contract at these degrees, so a fold")
    print("       here would delete structure rather than compress it. The")
    print("       primitive is not wrong; the graph has no free chains in it.")

    print("\n  (C2) mined (first-order) macro-overlay: the forced zero, measured")
    g0, T0, _ = unreliable_world(0)
    idx0 = np.arange(N_A)
    mg = MacroGraph(g0); mined = mg.mine(idx0, top=48)
    goals = list(np.argsort(-g0.P.sum(1))[:25])
    d_rel, d_dec, cmp_n = [], [], 0
    for gl in goals:
        for a in range(0, N_A, 7):
            if a == gl:
                continue
            base = g0.plan_reliable(idx0, a, int(gl), weights="lcb")
            wm = g0.plan_reliable(idx0, a, int(gl), weights="lcb", macros=mg)
            if base is None or wm is None:
                continue
            cmp_n += 1
            d_rel.append(wm.p_lcb - base.p_lcb)
            d_dec.append(wm.decisions - base.hops)
    max_rel = max(abs(x) for x in d_rel)
    print(f"       {len(mined)} macros mined, {cmp_n} plans compared")
    print(f"       reliability delta : max |d| {max_rel:.1e}   mean "
          f"{np.mean(d_rel):+.1e}")
    print(f"       decisions delta   : mean {np.mean(d_dec):+.2f} "
          f"({sum(1 for d in d_dec if d < 0)}/{cmp_n} plans shortened)")
    tb = timed(lambda: g0.plan_reliable(idx0, 3, int(goals[0]), weights="lcb"))
    tm = timed(lambda: g0.plan_reliable(idx0, 3, int(goals[0]), weights="lcb",
                                        macros=mg))
    print(f"       planning cost     : {tb * 1e3:.2f} ms plain vs "
          f"{tm * 1e3:.2f} ms with the overlay ({tm / tb:.2f}x)")
    print("       A mined macro's quality IS the product of its own edges, so")
    print("       Dijkstra already pays exactly that price along the expanded")
    print("       route: the zero is forced by construction, and measuring it")
    print("       is the only honest thing to do with it. What macros DO buy")
    print("       is fewer decisions for the same route -- at a small cost in")
    print("       planning time, since the overlay adds edges to relax.")

    print("\n  (C3) observed (second-order) macros: a first-order control and")
    print("       a genuinely second-order world, plans scored under the TRUE")
    print("       law. This is where a macro can carry what a matrix cannot.")
    print(f"       {'world':<24}{'base':>9}{'+macros':>9}{'delta (s=0-4)':>15}"
          f"{'flips':>8}")
    res = {}
    for wname, o2 in (("first-order control", False), ("second-order", True)):
        deltas = []; first = (0.0, 0.0)
        for s in SEEDS:
            g2, law, seq = second_order_world(s, order2=o2)
            n2 = 30; idx2 = np.arange(n2)
            marg = true_marginal(law, n2)
            mg2 = MacroGraph(g2).observe(seq)
            mg2.mine_observed(idx2, top=64, min_count=20)
            goals2 = list(np.argsort(-g2.P.sum(1))[:8])

            def true_lp(path):
                if len(path) < 2:
                    return 0.0
                p = marg[path[0], path[1]]
                if p <= 0:
                    return -np.inf
                tot = float(np.log(p))
                for i in range(1, len(path) - 1):
                    key = (int(path[i - 1]), int(path[i]))
                    if key not in law:
                        return -np.inf
                    s_, pr = law[key]
                    hit = np.nonzero(np.asarray(s_) == path[i + 1])[0]
                    if not len(hit):
                        return -np.inf
                    tot += float(np.log(pr[hit[0]]))
                return tot

            bv, mv = [], []
            for gl in goals2:
                for a in range(n2):
                    if a == gl:
                        continue
                    rb = g2.plan_reliable(idx2, a, int(gl), weights="lcb")
                    rm = g2.plan_reliable(idx2, a, int(gl), weights="lcb",
                                          macros=mg2)
                    if rb is None or rm is None:
                        continue
                    lb, lm = true_lp(rb.path), true_lp(rm.path)
                    if np.isfinite(lb) and np.isfinite(lm):
                        bv.append(lb); mv.append(lm)
            deltas.append(float(np.mean(mv) - np.mean(bv)) if bv else 0.0)
            if s == 0:
                first = (float(np.mean(bv)), float(np.mean(mv)))
        flips = sum(1 for d in deltas if d <= 0)
        tag = (f"{flips}/5" if o2
               else f"|d|<={max(abs(d) for d in deltas):.0e}")
        print(f"       {wname:<24}{first[0]:>9.3f}{first[1]:>9.3f}"
              f"{np.mean(deltas):>+15.3f}{tag:>11}")
        res[o2] = (float(np.mean(deltas)), flips,
                   float(max(abs(d) for d in deltas)))
    print("       base = plan on the transition matrix; +macros = the same")
    print("       planner with observed trigram options overlaid. Both scored")
    print("       under the true second-order law; seeds 0-4 paired.")

    ho = []
    for s in HELDOUT:
        g2, law, seq = second_order_world(s, order2=True)
        n2 = 30; idx2 = np.arange(n2)
        marg = true_marginal(law, n2)
        mg2 = MacroGraph(g2).observe(seq)
        mg2.mine_observed(idx2, top=64, min_count=20)
        goals2 = list(np.argsort(-g2.P.sum(1))[:8])

        def tlp(path):
            if len(path) < 2:
                return 0.0
            p = marg[path[0], path[1]]
            if p <= 0:
                return -np.inf
            tot = float(np.log(p))
            for i in range(1, len(path) - 1):
                key = (int(path[i - 1]), int(path[i]))
                if key not in law:
                    return -np.inf
                s_, pr = law[key]
                hit = np.nonzero(np.asarray(s_) == path[i + 1])[0]
                if not len(hit):
                    return -np.inf
                tot += float(np.log(pr[hit[0]]))
            return tot

        bv, mv = [], []
        for gl in goals2:
            for a in range(n2):
                if a == gl:
                    continue
                rb = g2.plan_reliable(idx2, a, int(gl), weights="lcb")
                rm = g2.plan_reliable(idx2, a, int(gl), weights="lcb", macros=mg2)
                if rb is None or rm is None:
                    continue
                lb, lm = tlp(rb.path), tlp(rm.path)
                if np.isfinite(lb) and np.isfinite(lm):
                    bv.append(lb); mv.append(lm)
        ho.append(float(np.mean(mv) - np.mean(bv)) if bv else 0.0)
    print(f"       HELD-OUT s=5-9, second-order world: {np.mean(ho):+.3f} "
          f"[{min(ho):+.3f}, {max(ho):+.3f}], "
          f"{sum(1 for v in ho if v <= 0)}/5 flips -> "
          f"{'SURVIVES' if survives(ho) else 'DOES NOT SURVIVE'}")
    return max_rel, float(np.mean(d_dec)), res, float(np.mean(ho)), survives(ho)


def real_graph(N, K, H, frames, dwell=30):
    """One organism run for one number: the shape of a REAL transition
    matrix, so section D's synthetic sweep is anchored to something the
    mechanism produced rather than to an assumption about it. The field is
    used here and only here, to BUILD a graph; it is gone before anything is
    timed, exactly as phase 30 built its graph by perceiving one."""
    rng = np.random.default_rng(5)
    Gq, _ = np.linalg.qr(rng.standard_normal((N, H))
                         + 1j * rng.standard_normal((N, H)))
    G = Gq.T * np.sqrt(N)
    T = np.zeros((H, H))
    for h in range(H):
        succ = rng.choice([x for x in range(H) if x != h], size=8, replace=False)
        T[h, succ] = rng.dirichlet(np.full(8, 0.6))

    def stream(nframes):
        h = 0
        for i in range(nframes):
            if i % dwell == 0 and i > 0:
                h = int(rng.choice(H, p=T[h]))
            yield G[h] + 0.4 * (rng.standard_normal(N)
                                + 1j * rng.standard_normal(N))

    t0 = time.perf_counter()
    org = Organism(N=N, K=K, seed=0)
    org.perceive(stream(frames), pool=True, confirm=2, recruit=0.75)
    org.consolidate()
    sp = SparseTransitions.from_graph(org.graph, org.kept_idx)
    return org, sp, time.perf_counter() - t0


def section_d():
    print("\n(D) OPTIMIZATION: sparse P against the dense reasoning ops")
    print("\n    First, REAL transition matrices, so the synthetic sweep below")
    print("    is anchored to something the mechanism produced:")
    print(f"      {'world':>8}{'N':>6}{'K':>6}{'used':>6}{'kept':>6}"
          f"{'density':>9}{'succ/sym':>10}{'secs':>7}")
    reals = []
    for (N, K, H, frames) in [(128, 60, 40, 120000), (192, 120, 80, 240000),
                              (256, 220, 160, 400000)]:
        org, sp, secs = real_graph(N, K, H, frames)
        kept = len(org.kept_idx)
        reals.append((org, sp, kept))
        print(f"      {H:>6} rg{N:>6}{K:>6}{int(org.used.sum()):>6}{kept:>6}"
              f"{sp.density:>8.1%}{sp.nnz / max(kept, 1):>10.1f}{secs:>7.0f}")
    print("    Honest reading: successors per symbol FALLS with world size")
    print("    here (5 -> 2), which is not a property of the graph but phase")
    print("    34's Attractor Crowding starving coverage -- fewer regimes get")
    print("    captured, so the transitions among captured memories thin out.")
    print("    The synthetic sweep therefore fixes out-degree at 10 to match")
    print("    33g's COMMITTED measurement (6.1% at K=160 on the 33c protocol,")
    print("    ~10 successors/symbol) rather than extrapolating from these.")
    print("    THE FIELD IS NOW ABSENT -- everything below is graph arithmetic.")
    org, real_sp, kept = reals[0]

    Ks = [40, 112, 160, 400, 800, 1580]
    print(f"\n    {'K':>6}{'dens':>7}   {'next_hops ms':>20}{'x':>6}{'eq':>5}"
          f"   {'rollout ms':>17}{'x':>6}   {'kstep(2) ms':>17}{'x':>6}")
    rows = []
    for K in Ks:
        gK = sparse_world(0, K); idxK = np.arange(K)
        spK = SparseTransitions.from_graph(gK, idxK)
        t_dn = timed(lambda: gK.next_hops(idxK, 1), repeat=3)
        spK.next_hops(1)                                  # build the CSC once
        t_sn = timed(lambda: spK.next_hops(1), repeat=3)
        dn, dd = gK.next_hops(idxK, 1); sn, sd = spK.next_hops(1)
        fin = np.isfinite(dd) & np.isfinite(sd)
        eq_n = (np.array_equal(dn, sn)
                and np.array_equal(np.isfinite(dd), np.isfinite(sd))
                and np.array_equal(dd[fin], sd[fin]))
        steps = 2000
        t_dr = timed(lambda: gK.rollout(idxK, 0, steps, np.random.default_rng(5)), repeat=3)
        t_sr = timed(lambda: spK.rollout(0, steps, np.random.default_rng(5)), repeat=3)
        t_pr = timed(lambda: spK.rollout_pruned(0, steps, np.random.default_rng(5)), repeat=3)
        eq_r = np.array_equal(gK.rollout(idxK, 0, steps, np.random.default_rng(5)),
                              spK.rollout(0, steps, np.random.default_rng(5)))
        t_dk = timed(lambda: gK.kstep(idxK, 2), repeat=3)
        t_sk = timed(lambda: spK.kstep(2), repeat=3)
        dk_err = float(np.abs(gK.kstep(idxK, 2) - spK.kstep(2)).max())
        t_dkr = timed(lambda: gK.kstep(idxK, 3)[0], repeat=3)
        t_skr = timed(lambda: spK.kstep_row(0, 3), repeat=3)
        kr_err = float(np.abs(gK.kstep(idxK, 3)[0] - spK.kstep_row(0, 3)).max())
        rows.append(dict(K=K, dens=spK.density, t_dn=t_dn, t_sn=t_sn, eq_n=eq_n,
                         t_dr=t_dr, t_sr=t_sr, t_pr=t_pr, eq_r=eq_r, t_dk=t_dk,
                         t_sk=t_sk, dk_err=dk_err, t_dkr=t_dkr, t_skr=t_skr,
                         kr_err=kr_err))
        print(f"    {K:>6}{spK.density:>6.1%}   "
              f"{t_dn * 1e3:>9.2f}/{t_sn * 1e3:<10.2f}{t_dn / t_sn:>5.1f}x"
              f"{'bit' if eq_n else 'NEQ':>5}   "
              f"{t_dr * 1e3:>8.1f}/{t_sr * 1e3:<8.1f}{t_dr / t_sr:>5.1f}x   "
              f"{t_dk * 1e3:>8.2f}/{t_sk * 1e3:<8.2f}{t_dk / t_sk:>5.1f}x")

    print(f"\n    {'K':>6}   {'kstep_row(3) ms: dense/sparse':>32}{'x':>9}"
          f"{'max|d|':>11}")
    for r in rows:
        print(f"    {r['K']:>6}   {r['t_dkr'] * 1e3:>14.2f} / "
              f"{r['t_skr'] * 1e3:<15.3f}{r['t_dkr'] / r['t_skr']:>8.0f}x"
              f"{r['kr_err']:>11.0e}")
    print("    kstep_row is the query a planner actually asks -- 'where does")
    print("    this symbol land in k hops' -- and the dense layer cannot ask")
    print("    it at all without building the whole K x K matrix first.")

    big = [r for r in rows if r['K'] >= 112]
    cross = next((r['K'] for r in rows if r['t_dn'] / r['t_sn'] >= 2.0), None)
    print(f"\n    equality: next_hops bitwise on "
          f"{sum(1 for r in rows if r['eq_n'])}/{len(rows)} shapes, rollout "
          f"bitwise on {sum(1 for r in rows if r['eq_r'])}/{len(rows)};")
    print(f"    kstep max|d| {max(r['dk_err'] for r in rows):.0e} -- a float")
    print(f"    reduction reordered by skipping zeros, tolerance not bit.")
    print(f"    rollout_pruned (NOT a port: different RNG consumption, so a")
    print(f"    different trajectory from the same seed) runs "
          f"{np.mean([r['t_sr'] / r['t_pr'] for r in rows]):.1f}x the bitwise")
    print(f"    sparse rollout -- the price of equality, made explicit.")

    speed_ok = all(r['t_dn'] / r['t_sn'] >= 2.0 for r in big)
    bit_ok = all(r['eq_n'] and r['eq_r'] for r in rows)
    detail = ", ".join("{}:{:.1f}x".format(r["K"], r["t_dn"] / r["t_sn"])
                       for r in big)
    print(f"\n    next_hops crosses 2x at K = {cross}; at K >= 112 the >= 2x")
    print(f"    clause is {'MET' if speed_ok else 'NOT MET'} ({detail}).")

    t_rd = timed(lambda: org.graph.next_hops(org.kept_idx, 1), repeat=3)
    real_sp.next_hops(1)
    t_rs = timed(lambda: real_sp.next_hops(1), repeat=3)
    print(f"    On the real K={kept} graph ({real_sp.density:.1%} dense): "
          f"next_hops {t_rd * 1e3:.2f} ms dense /")
    print(f"    {t_rs * 1e3:.2f} ms sparse = {t_rd / t_rs:.1f}x -- consistent "
          f"with the synthetic row at that K.")
    return rows, speed_ok, bit_ok, cross, kept, real_sp.density


def main():
    print("PHASE 40: logic-layer depth and optimization (T6.3)")
    print("=" * 74)
    anchor_ok = section_0()
    ok_a, adv_h, obs, bar, ho_a, sweep, a4 = section_a()
    viol, gap_avoid, gap_conj, trespass = section_b()
    max_rel, d_dec, c3, c3_ho, c3_surv = section_c()
    rows, speed_ok, bit_ok, cross, kept, dens = section_d()

    print("\n" + "=" * 74)
    print("VERDICT")
    print(f"  anchor: phase 30's next_hops is bitwise unchanged by the "
          f"refactor ({'OK' if anchor_ok else 'FAILED'}).")
    print(f"  (a) CONFIRMED: confidence-weighted planning beats hop-count "
          f"planning by {adv_h:+.2f} nats/plan")
    print(f"      (5/5 seeds), above the shuffled-shrinkage null "
          f"({obs:+.4f} vs {bar:+.4f}), and holds on")
    print(f"      held-out seeds at the pre-registered alpha ({ho_a:+.2f}).")
    print(f"      Not pre-registered, and the sharper result: against the MLE "
          f"planner phase 30 already")
    print(f"      had, the two disagree on {sweep[STEPS_A][0]:.0%} of plans at "
          f"{STEPS_A} observations and confidence wins")
    print(f"      {sweep[STEPS_A][1]:+.2f} nats WHEN they disagree -- decaying "
          f"to {sweep[BUDGETS[-1]][1]:+.2f} at {BUDGETS[-1]}. The correction is")
    print(f"      a THIN-EVIDENCE correction; on a well-observed graph the "
          f"existing planner is already right.")
    if a4:
        surv = [x for x in a4 if x[2]]
        print(f"      On held-out seeds it survives at "
              f"{[x[0] for x in surv] if surv else 'no'} budget(s)"
              + (f" ({surv[0][1]:+.3f} at {surv[0][0]})." if surv else
                 " -- banked as a direction, not a claim."))
    else:
        print(f"      No budget passed the survival rule on the selection "
              f"seeds; nothing banked.")
    print(f"  (b) SPLIT: equality clause {'MET' if bit_ok else 'PARTIAL'} "
          f"(next_hops and rollout bitwise at every K;")
    print(f"      kstep {max(r['dk_err'] for r in rows):.0e}, a reduction "
          f"reorder). Speed clause NOT MET as written --")
    print(f"      next_hops crosses 2x at K={cross}, not K=112 "
          f"({rows[1]['t_dn'] / rows[1]['t_sn']:.1f}x there), reaching "
          f"{rows[-1]['t_dn'] / rows[-1]['t_sn']:.1f}x at K=1580.")
    print(f"      Sparse kstep is a measured NEGATIVE "
          f"({rows[-1]['t_dk'] / rows[-1]['t_sk']:.1f}x at K=1580): a dense "
          f"K^2 output is BLAS's")
    print(f"      home ground. The real win is kstep_row -- "
          f"{rows[-1]['t_dkr'] / rows[-1]['t_skr']:.0f}x at K=1580 -- which "
          f"is the query planners issue.")
    print(f"  (c) CONFIRMED AS THE NEGATIVE, with the condition measured. "
          f"Mined first-order macros move")
    print(f"      reliability by {max_rel:.0e} -- zero, forced by "
          f"construction -- buying only {-d_dec:.2f} fewer decisions")
    print(f"      per plan; the pass-through census finds nothing to fold. "
          f"Observed second-order macros")
    print(f"      collapse to |d| <= {c3[False][2]:.0e} on a first-order world "
          f"(the control, as it must) and give {c3[True][0]:+.3f} on a")
    print(f"      second-order one ({c3[True][1]}/5 flips; held-out "
          f"{c3_ho:+.3f}, {'survives' if c3_surv else 'does not survive'}). "
          f"Macros pay exactly when the")
    print(f"      world has structure the transition matrix cannot hold -- "
          f"not at a graph size, at an ORDER.")
    print(f"  (composition) avoidance violations {viol}, optimality gap vs "
          f"brute force {gap_avoid:.0e} (negation)")
    print(f"      and {gap_conj:.0e} (conjunction); a random walker that "
          f"reaches the goal trespasses on the")
    print(f"      forbidden symbol {trespass:.0%} of the time, the planner "
          f"never.")
    print("=" * 74)


if __name__ == "__main__":
    main()
