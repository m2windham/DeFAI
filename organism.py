"""
ORGANISM (Phase 1 consolidation): perceive -> form memories -> learn the world's
SEQUENTIAL structure -> recall by generating learned trajectories.

Advance over learn_world.py: the world now has ORDER. Regimes follow a structured
transition graph, not random switching. The organism must learn BOTH the memories
(what the regimes are) AND the transition structure (which follows which), then in
recall -- with no input -- GENERATE sequences that follow the learned structure.
That is the jump from wandering memory to imagining plausible trajectories.

Pipeline (one module, clean API):
  Organism.perceive(stream)  -> competitive memory formation + Hebbian transition
                                 learning across time.
  Organism.consolidate()     -> merge duplicate memories, drop unused slots.
  Organism.recall(steps)     -> autonomous sequence generation: fatigue releases the
                                 current memory, transition prior P[cur,:] biases the
                                 next, field locks onto a likely successor.

Honesty: we score the generated sequences against the TRUE transition graph and
against a random-successor BASELINE.

ARCHITECTURE (logic/language separation): the module is split into two systems
with a narrow interface, mirroring the neuroscience finding that the brain's
language network and its logical-reasoning system are functionally decoupled
(aphasia patients keep full relational reasoning; language regions stay silent
during logic tasks -- MIT/McGovern 2026):

  PERCEPTION ("language"):  Organism.perceive -- field settling, saccade
      segmentation, recruitment, pooling, fusion, recycling. Decides WHAT was
      just seen. Owns all slot-lifecycle state (prov/hits/age/nvis).
  LOGIC ("reasoning"):      TransitionGraph -- the relational structure
      (which follows which). Knows nothing about fields, overlaps, or bars.
  BOUNDARY:                 EventBoundary -- perception emits only committed,
      confidence-gated symbol events across it; identity changes (fusion,
      recycling) are explicit notifications, not in-place matrix surgery.

  The payoff is the aphasia-lesion property: the perceptual front-end can be
  swapped or damaged without touching the learned world graph, recognition
  errors cannot silently rewrite relational knowledge (they are stopped at the
  boundary's confidence gate), and each layer is testable alone -- the graph
  on clean synthetic symbol streams, perception on ground-truth coverage.
  Slots are storage; SYMBOLS are identity. `SymbolRegistry` (T3.3, opt-in via
  Organism(symbols=True)) mints an ID the first time a slot crosses the
  boundary and carries it through fusion, recycling, consolidation and
  save/load, driven entirely by the boundary's existing commit/remap/
  invalidate notifications. Downstream scripts may still index org.P by slot
  -- the registry is additive and observational -- but `slot_of(sid)` is the
  index that stays correct across slot churn.
"""

import collections
import itertools
import math
import os

import numpy as np

import fastpath

_SQRT2 = math.sqrt(2.0)


def normalize(v, norm):
    return v / (np.linalg.norm(v) + 1e-9) * norm


class TransitionGraph:
    """LOGIC LAYER: relational structure over symbols, decoupled from how
    symbols are perceived. The only mutations are the four interface ops
    below -- perception never touches the matrix directly, so every way the
    world-graph can change is enumerable and auditable here."""

    def __init__(self, K):
        self.P = np.zeros((K, K))   # transition counts between symbols

    def observe(self, a, b, decay=0.0):
        """One committed transition a -> b. `decay` is exponential forgetting
        applied once per observed transition (synaptic decay, phase 11)."""
        if decay > 0.0:
            self.P *= (1.0 - decay)
        self.P[a, b] += 1

    def merge(self, keep, drop):
        """Two symbols turned out to be the same pattern (slot fusion):
        `drop`'s relational history folds into `keep`, then `drop` is erased."""
        self.P[keep] += self.P[drop]; self.P[:, keep] += self.P[:, drop]
        self.retire(drop)

    def retire(self, idx):
        """Symbol(s) recycled (use it or lose it): dead structure carries no
        relations. `idx` may be an int or a boolean mask."""
        self.P[idx] = 0; self.P[:, idx] = 0

    def fold(self, into, frm):
        """Consolidation-time duplicate folding: add `frm`'s transitions into
        `into` WITHOUT erasing `frm` -- callers snapshot/restore raw counts
        around consolidation (see phase 19), so folding must be additive."""
        self.P[into] += self.P[frm]; self.P[:, into] += self.P[:, frm]

    def normalized(self, idx):
        """Row-normalized transition probabilities restricted to symbols
        `idx` -- the prior the recall dynamics consume."""
        Pm = self.P[np.ix_(idx, idx)]
        return Pm / (Pm.sum(1, keepdims=True) + 1e-9)

    # ---- PHASE 30: reasoning ops -- pure symbol-space, no field anywhere ----
    def kstep(self, idx, k):
        """Multi-step inference: P(symbol after k hops | symbol now), chained
        from one-step knowledge. These are relational conclusions the organism
        never directly observed -- transitivity, computed without the field."""
        return np.linalg.matrix_power(self.normalized(idx), k)

    def rollout(self, idx, start, steps, rng):
        """Symbolic imagination: sample a trajectory purely in symbol space.
        No settling, no attractors, no habituation -- thinking without the
        'language' system. Stops early at a symbol with no outgoing mass."""
        Pn = self.normalized(idx)
        seq = []; cur = int(start)
        for _ in range(steps):
            row = Pn[cur]; s = row.sum()
            if s <= 0:
                break
            cur = int(rng.choice(len(row), p=row / s))
            seq.append(cur)
        return np.array(seq, int)

    @staticmethod
    def dijkstra(Q, goal, eps=1e-12):
        """Backward Dijkstra over an edge-QUALITY matrix `Q` (higher is
        better; Q[i,j] is a probability-like score for the hop i -> j, and
        Q[i,j] <= eps means "no such edge"). Edge cost is -log Q, so the
        tree returned maximizes the PRODUCT of qualities along the path.

        Factored out of `next_hops` in phase 40 (T6.3) because every planner
        this layer grows differs from the others only in how Q is built --
        point-estimate transitions, confidence lower bounds, unit hop costs,
        a graph with symbols forbidden, a graph with macro-options added.
        One algorithm, several notions of "best", and the sparse port has a
        single dense implementation to be equal to."""
        n = Q.shape[0]
        W = -np.log(Q + eps)
        dist = np.full(n, np.inf); dist[goal] = 0.0
        nxt = np.full(n, -1, int); nxt[goal] = goal
        done = np.zeros(n, bool)
        for _ in range(n):
            u = int(np.argmin(np.where(done, np.inf, dist)))
            if not np.isfinite(dist[u]) or done[u]:
                break
            done[u] = True
            relax = (Q[:, u] > eps) & (dist[u] + W[:, u] < dist)
            nxt[relax] = u
            dist[relax] = dist[u] + W[relax, u]
        return nxt, dist

    def next_hops(self, idx, goal, eps=1e-12):
        """Planning: for every symbol, the first hop on the most-probable
        path to `goal` (Dijkstra on -log transition probability, run backward
        from the goal). Returns (nxt, dist): nxt[i] = -1 if goal unreachable
        from i, dist[i] = -log P(best path i -> goal)."""
        return self.dijkstra(self.normalized(idx), goal, eps)

    def plan(self, idx, a, b):
        """Most-probable path a -> b as a list of symbols (inclusive), or
        None if b is unreachable from a."""
        nxt, _ = self.next_hops(idx, b)
        return _trace(nxt, a, b)

    # ---- PHASE 40 (T6.3): reasoning DEPTH -- still pure symbol space -------
    # Everything below runs on counts alone. No field, no overlaps, no
    # embeddings: phase 30's isolation discipline is the precondition for the
    # whole layer, not a property of the three ops it happened to ship with.

    @staticmethod
    def _z_for(alpha):
        """Upper-`alpha` quantile of the standard normal, by bisection on
        erfc. Dependency-free on purpose -- scipy is optional in this project
        (it rides along with numba) and the logic layer must not need it."""
        lo, hi = 0.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * math.erfc(mid / _SQRT2) > alpha:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def confidence(self, idx, alpha=0.05, mode="wilson"):
        """EDGE CONFIDENCE: an evidence-aware edge probability, where
        `normalized` gives the evidence-blind point estimate.

        The count matrix carries EVIDENCE as well as frequency and
        `normalized` discards it. A symbol visited once, whose one successor
        was j, gets Pn = 1.0 -- a zero-cost edge that Dijkstra will happily
        route the entire world through -- while an edge seen 900 times out of
        1000 visits gets 0.9 and scores worse. Planning on the point estimate
        is therefore biased toward the least-observed corner of the graph,
        which is exactly where the model is least entitled to be believed.

        'wilson' returns the Wilson score interval's LOWER END at level
        `alpha`: a 1/1 edge falls to ~0.21 at alpha=0.05 while a 900/1000
        edge barely moves, and the result never exceeds the point estimate.
        'laplace' returns the add-one posterior mean -- the milder classical
        alternative, and NOT a lower bound: it shrinks confident edges down
        but also lifts rare ones up, which is correct for a posterior mean
        and is why only the wilson arm is pinned as a bound (harness 12).

        Unobserved edges stay at exactly 0.0 under both -- smoothing must not
        invent hops the organism never saw, or the planner will route through
        transitions that never happened. Rows are NOT renormalized: these are
        per-edge reliabilities, not a distribution, and planners consume them
        as such."""
        C = np.asarray(self.P[np.ix_(idx, idx)], float)
        n = C.sum(1, keepdims=True)
        seen = C > 0
        safe_n = np.where(n > 0, n, 1.0)
        if mode == "laplace":
            return np.where(seen, (C + 1.0) / (safe_n + C.shape[1]), 0.0)
        if mode != "wilson":
            raise ValueError(f"unknown confidence mode {mode!r}")
        z = self._z_for(alpha); z2 = z * z
        p = C / safe_n
        centre = p + z2 / (2 * safe_n)
        half = z * np.sqrt(p * (1 - p) / safe_n + z2 / (4 * safe_n * safe_n))
        lo = (centre - half) / (1 + z2 / safe_n)
        return np.where(seen, np.maximum(lo, 0.0), 0.0)

    def edge_quality(self, idx, weights="mle", alpha=0.05, mode="wilson"):
        """The edge-quality matrix a planner runs on -- the one axis that
        separates phase 40's planners from phase 30's:

          'mle'  row-normalized counts. Phase 30's planner: most-probable
                 path, evidence ignored.
          'lcb'  `confidence` lower bounds: most-RELIABLE path.
          'hops' equal quality on every observed edge, so every hop costs
                 exactly 1.0 and Dijkstra minimizes hop count. (exp(-1), not
                 1.0, so the cost stays positive -- Dijkstra is only correct
                 with non-negative weights.)"""
        if weights == "mle":
            return self.normalized(idx)
        if weights == "lcb":
            return self.confidence(idx, alpha, mode)
        if weights == "hops":
            return np.where(np.asarray(self.P[np.ix_(idx, idx)]) > 0,
                            math.exp(-1.0), 0.0)
        raise ValueError(f"unknown planner weights {weights!r}")

    def path_report(self, idx, path, alpha=0.05, mode="wilson", decisions=None):
        """Score a concrete path: the product of its edge probabilities as a
        point estimate AND as a lower bound, plus its weakest link and how
        many times that link was actually observed.

        This is the "report path reliability, not just shortest hops" half of
        the op. A two-hop plan through an edge seen once is not a better
        answer than a four-hop plan through edges seen a thousand times, and
        a caller that is only handed a hop count cannot tell the difference."""
        Pn = self.normalized(idx)
        L = self.confidence(idx, alpha, mode)
        C = np.asarray(self.P[np.ix_(idx, idx)], float)
        p_mle = p_lcb = 1.0
        weakest, weakest_n = -1, np.inf
        for a, b in zip(path[:-1], path[1:]):
            p_mle *= float(Pn[a, b]); p_lcb *= float(L[a, b])
            if L[a, b] < weakest_n:
                weakest, weakest_n = (int(a), int(b)), float(L[a, b])
        obs = np.inf if weakest == -1 else float(C[weakest[0], weakest[1]])
        hops = len(path) - 1
        return PathReport(tuple(int(s) for s in path), hops,
                          hops if decisions is None else int(decisions),
                          p_mle, p_lcb, weakest, obs)

    @staticmethod
    def _forbid(Q, avoid):
        """The negated half of a compositional goal: delete forbidden
        symbols from the graph the planner sees. Deleting the symbol (row AND
        column) rather than scoring it badly is what makes "without passing
        B" a constraint instead of a preference."""
        if avoid is None or len(avoid) == 0:
            return Q
        Q = Q.copy()
        av = np.asarray(sorted({int(v) for v in avoid}), int)
        Q[av, :] = 0.0; Q[:, av] = 0.0
        return Q

    def plan_reliable(self, idx, a, b, weights="lcb", alpha=0.05,
                      mode="wilson", avoid=(), macros=None, eps=1e-12):
        """Phase 40's planner: `plan` generalized along three axes -- which
        notion of edge quality (`weights`), which symbols are off limits
        (`avoid`), and whether multi-hop macro-options may be taken
        (`macros`, a MacroGraph). Returns a PathReport scored on the plain
        transition statistics whatever the planner optimized, so arms are
        comparable, or None if `b` is unreachable under the constraints."""
        a, b = int(a), int(b)
        if a in set(int(v) for v in avoid) or b in set(int(v) for v in avoid):
            return None
        Q = self._forbid(self.edge_quality(idx, weights, alpha, mode), avoid)
        Qm = None if macros is None else macros.overlay(Q, avoid)
        nxt, _ = self.dijkstra(Q if Qm is None else np.maximum(Q, Qm), b, eps)
        path = _trace(nxt, a, b)
        if path is None:
            return None
        decisions = len(path) - 1                # choices BEFORE expansion:
        if Qm is not None:                       # the only thing a first-order
            path = macros.expand(path, Q, Qm)    # macro can actually move
        return self.path_report(idx, path, alpha, mode, decisions)

    def plan_visit(self, idx, a, goals, ordered=False, weights="lcb",
                   alpha=0.05, mode="wilson", avoid=(), eps=1e-12):
        """CONJUNCTIVE goals: a walk from `a` that visits every symbol in
        `goals` -- in the given order if `ordered`, otherwise in whichever
        order is cheapest -- while respecting `avoid`.

        Segment costs are additive and the graph is memoryless, so stitching
        per-segment optimal paths is optimal for a FIXED order, and the order
        search is exhaustive; a goal already crossed incidentally by an
        earlier segment is dropped from the remaining itinerary rather than
        re-visited, which is where the cheap wins are. Honest scope: the
        search is over orders of the goals, so `goals` must stay small (6 is
        the cap -- 720 orders); this is composition, not a TSP solver."""
        a = int(a); goals = [int(g) for g in goals]
        banned = {int(v) for v in avoid}
        if a in banned or any(g in banned for g in goals):
            return None
        if len(goals) > 6:
            raise ValueError("plan_visit exhausts goal orders; keep len(goals) <= 6")
        Q = self._forbid(self.edge_quality(idx, weights, alpha, mode), avoid)
        trees = {g: self.dijkstra(Q, g, eps)[0] for g in set(goals)}
        best = None
        for order in itertools.permutations(sorted(set(goals))):
            walk, cur, ok = [a], a, True
            for g in order:
                if g in walk:                    # already crossed on the way
                    continue
                leg = _trace(trees[g], cur, g)
                if leg is None:
                    ok = False; break
                walk += leg[1:]; cur = g
            if not ok:
                continue
            rep = self.path_report(idx, walk, alpha, mode)
            if best is None or rep.p_lcb > best.p_lcb:
                best = rep
        return best

    def passthrough_census(self, idx, min_count=1):
        """TEMPORAL ABSTRACTION, the `fold` reading: which symbols are pure
        pass-throughs -- exactly one observed predecessor and exactly one
        observed successor -- so that a -> m -> b could be contracted into a
        single edge by folding m away.

        Returns the list of (pred, m, succ) triples. Run this BEFORE building
        anything: 33h's redundancy census called the fold verdict there
        before a single fold ran, and the same question applies here. A graph
        with no pass-throughs has no free contraction, and every byte a
        contraction saves then costs structure."""
        C = np.asarray(self.P[np.ix_(idx, idx)], float)
        live = C > min_count - 1e-12
        found = []
        for m in range(C.shape[0]):
            preds = np.nonzero(live[:, m])[0]
            succs = np.nonzero(live[m])[0]
            if len(preds) == 1 and len(succs) == 1 and preds[0] != m != succs[0]:
                found.append((int(preds[0]), int(m), int(succs[0])))
        return found

    def contracted(self, idx, chains=None, min_count=1):
        """A COPY of this graph with pass-through symbols folded away, built
        with the layer's own additive primitive (`fold` then `retire`, on the
        copy -- the live graph is never mutated, so a contraction can never
        cost the organism a symbol it might still perceive). Returns
        (graph, surviving_index) where `surviving_index` selects the rows of
        the contracted graph that are still live."""
        sub = TransitionGraph(len(idx))
        sub.P = np.array(self.P[np.ix_(idx, idx)], float)
        gone = set()
        for pred, m, succ in (self.passthrough_census(idx, min_count)
                              if chains is None else chains):
            if m in gone or pred in gone or succ in gone:
                continue
            sub.fold(pred, m)                    # m's relations join pred's
            sub.P[pred, m] = 0.0; sub.P[m, pred] = 0.0
            sub.retire(m)
            gone.add(m)
        keep = np.array([i for i in range(len(idx)) if i not in gone], int)
        return sub, keep


PathReport = collections.namedtuple(
    "PathReport", "path hops decisions p_mle p_lcb weakest weakest_n")
PathReport.__doc__ = """A plan and what it is actually worth (phase 40).

`p_mle` is the product of point-estimate transition probabilities along the
path, `p_lcb` the same product over confidence lower bounds; `weakest` is the
(from, to) edge with the lowest lower bound and `weakest_n` how many times
that edge was observed. Two plans with the same `hops` can differ by orders
of magnitude in `p_lcb`, which is the entire reason this type exists.
`decisions` counts the hops the AGENT chooses -- equal to `hops` unless
macro-options collapsed several of them into one."""


def _trace(nxt, a, b):
    """Walk a Dijkstra next-hop tree from `a` to `b`, or None if unreached."""
    a, b = int(a), int(b)
    if nxt[a] < 0:
        return None
    path = [a]
    while path[-1] != b:
        path.append(int(nxt[path[-1]]))
        if len(path) > len(nxt):        # defensive: no cycles possible,
            return None                  # but never loop forever
    return path


class MacroGraph:
    """TEMPORAL ABSTRACTION over a TransitionGraph (phase 40, T6.3).

    A macro-edge is an OPTION: a multi-hop route the world runs often enough
    to be worth naming, collapsed into a single decision. There are two ways
    to get one and the difference between them is the whole experiment.

      `mine`      derives macros from the transition matrix itself. A mined
                  macro's quality is the PRODUCT of its constituent one-step
                  probabilities, so it cannot beat planning over its own
                  parts: Dijkstra already pays exactly that product for the
                  expanded route. This arm exists to MEASURE that zero, not
                  to beat it -- the first-order graph already knows
                  everything a first-order macro could tell it.
      `observe`   counts committed symbol n-grams directly. P(c | b, a) is
                  not P(c | b) unless the world is first-order, so an
                  observed macro CAN carry structure the transition matrix is
                  structurally unable to represent -- and when the world is
                  first-order it collapses back onto the product, measurably.

    Nothing here mutates the graph. `TransitionGraph.fold` is the additive
    primitive for collapsing two symbols that turned out to be the SAME
    symbol; naming a route between symbols that stay distinct is a different
    operation, and doing it as a planning-time overlay keeps the four
    auditable graph mutators intact. (The fold reading of temporal
    abstraction -- contracting pass-through symbols -- lives on the graph as
    `passthrough_census` / `contracted`, and is measured separately.)"""

    def __init__(self, graph=None):
        self.graph = graph
        self.macros = []                 # (src, dst, path tuple, quality)
        self.tri = {}                    # (a, b) -> {c: count}, observed
        self.bi = {}                     # (a, b) -> count, observed

    # ---- construction ---------------------------------------------------
    def mine(self, idx, top=32, max_len=3, alpha=0.05, mode="wilson",
             weights="lcb"):
        """First-order macros: the `top` highest-traffic multi-hop routes,
        scored as the product of their edge qualities. Traffic is the Markov
        estimate -- visits to the source times the chained probabilities --
        so "frequently traversed" is read off the matrix, not remembered."""
        g = self.graph
        C = np.asarray(g.P[np.ix_(idx, idx)], float)
        Pn = g.normalized(idx)
        Q = g.edge_quality(idx, weights, alpha, mode)
        visits = C.sum(1)
        n = C.shape[0]
        cand = {}
        for a in range(n):
            for m in np.nonzero(C[a])[0]:
                flow2 = visits[a] * Pn[a, m]
                for b in np.nonzero(C[m])[0]:
                    if b == a:
                        continue
                    flow = flow2 * Pn[m, b]
                    key = (a, int(b))
                    q = float(Q[a, m] * Q[m, b])
                    prev = cand.get(key)
                    if prev is None or q > prev[1]:
                        cand[key] = ((a, int(m), int(b)), q, flow)
                    if max_len >= 3:
                        for c in np.nonzero(C[b])[0]:
                            if c in (a, m):
                                continue
                            key3 = (a, int(c))
                            q3 = q * float(Q[b, c])
                            f3 = flow * Pn[b, c]
                            p3 = cand.get(key3)
                            if p3 is None or q3 > p3[1]:
                                cand[key3] = ((a, int(m), int(b), int(c)), q3, f3)
        ranked = sorted(cand.values(), key=lambda r: -r[2])[:top]
        self.macros = [(p[0], p[-1], p, q) for p, q, _ in ranked]
        return self.macros

    def observe(self, seq):
        """Count committed symbol bigrams and trigrams from one sequence.
        The logic layer already emits this sequence at the EventBoundary; a
        macro layer is what you get by remembering pairs of hops instead of
        single ones. Label-free, field-free, and additive across calls."""
        seq = [int(s) for s in seq]
        for a, b in zip(seq, seq[1:]):
            self.bi[(a, b)] = self.bi.get((a, b), 0) + 1
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            self.tri.setdefault((a, b), {})
            self.tri[(a, b)][c] = self.tri[(a, b)].get(c, 0) + 1
        return self

    def mine_observed(self, idx, top=32, min_count=8, alpha=0.05,
                      mode="wilson"):
        """Second-order macros: for each observed pair (a, b), the successor
        c that actually follows it, with the macro's quality taken from the
        SECOND-ORDER conditional P(c | a, b) rather than the matrix's
        P(c | b). Both factors get the same Wilson treatment the one-step
        planner uses, so a trigram seen three times cannot outrank a bigram
        seen a thousand."""
        g = self.graph
        L = g.confidence(idx, alpha, mode)
        z = g._z_for(alpha)
        out = []
        for (a, b), succ in self.tri.items():
            n_ab = sum(succ.values())
            if n_ab < min_count or a >= L.shape[0] or b >= L.shape[0]:
                continue
            c, c_n = max(succ.items(), key=lambda kv: kv[1])
            if c >= L.shape[0] or c == a:
                continue
            q2 = _wilson_lo(c_n, n_ab, z)
            out.append(((a, b, int(c)), float(L[a, b]) * q2, n_ab))
        out.sort(key=lambda r: -r[2])
        self.macros = [(p[0], p[-1], p, q) for p, q, _ in out[:top]]
        return self.macros

    # ---- planning-time use ----------------------------------------------
    def overlay(self, Q, avoid=()):
        """The macro edges as a quality matrix the same shape as `Q`, so a
        planner takes `maximum(Q, overlay)` and needs no special cases: a
        macro is just an edge that happens to expand into several."""
        banned = {int(v) for v in avoid}
        M = np.zeros_like(Q)
        for src, dst, path, q in self.macros:
            if banned & set(path) or src == dst:
                continue
            M[src, dst] = max(M[src, dst], q)
        return M

    def expand(self, path, Q, Qm, tol=0.0):
        """Rewrite a plan's macro-hops back into the symbols they stand for,
        so the returned plan is always a real walk on the base graph."""
        if not self.macros:
            return path
        by_pair = {}
        for src, dst, mpath, q in self.macros:
            cur = by_pair.get((src, dst))
            if cur is None or q > cur[1]:
                by_pair[(src, dst)] = (mpath, q)
        out = [int(path[0])]
        for u, v in zip(path[:-1], path[1:]):
            hit = by_pair.get((int(u), int(v)))
            if hit is not None and Qm[u, v] > Q[u, v] + tol:
                out += [int(s) for s in hit[0][1:]]
            else:
                out.append(int(v))
        return out


def _wilson_lo(c, n, z):
    """Wilson score lower bound for c successes out of n, at quantile z."""
    if n <= 0 or c <= 0:
        return 0.0
    p = c / n; z2 = z * z
    lo = ((p + z2 / (2 * n) - z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)))
          / (1 + z2 / n))
    return max(lo, 0.0)


class SparseTransitions:
    """CSR/CSC view of the logic layer's transition matrix, for field-free
    reasoning at scale (phase 40, T6.3 lever iv).

    33g measured P at 6.1% density at K=160 -- about ten live successors per
    symbol, and out-degree does not grow with K, so the matrix gets emptier
    the bigger the organism is. Every phase-30 reasoning op nonetheless walks
    the dense K x K array: `normalized` gathers and divides K^2 entries and
    `next_hops` takes K^2 logarithms before it relaxes a single edge. This
    view pays one O(K^2) scan -- or none at all, when the caller already
    holds a compressed store, which 33g's persistence path does
    (`from_csr`) -- and then works over nonzeros only.

    EQUIVALENCE IS THE POINT, so the ops here reproduce the dense ones'
    arithmetic rather than merely approximating it:

      `next_hops`, `plan`, `rollout`   BITWISE identical to TransitionGraph's,
          including the RNG consumption order, whenever P holds integer
          counts (row sums of exact integers are order-independent, so
          summing only the nonzeros lands on the same float). With p_decay >
          0 the counts stop being integers and the row sums become
          order-dependent; equality is then a tolerance, not a bit.
      `kstep`, `kstep_row`   NOT bitwise, and cannot be: skipping zeros
          reorders a float reduction. Held to a tolerance instead, per the
          project's standing rule for ports."""

    def __init__(self, data, indices, indptr, n):
        self.data = np.asarray(data, float)
        self.indices = np.asarray(indices, np.int64)
        self.indptr = np.asarray(indptr, np.int64)
        self.n = int(n)
        self.rowsum = np.add.reduceat(
            np.concatenate([self.data, [0.0]]), self.indptr[:-1])
        empty = np.diff(self.indptr) == 0
        self.rowsum = np.where(empty, 0.0, self.rowsum)
        self._pn = self.data / (self.rowsum[self._rows()] + 1e-9)
        self._csc = None

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_graph(cls, graph, idx):
        return cls.from_dense(graph.P[np.ix_(idx, idx)])

    @classmethod
    def from_dense(cls, P):
        P = np.asarray(P, float)
        n = P.shape[0]
        keep = P > 0
        counts = keep.sum(1)
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(counts, out=indptr[1:])
        rows, cols = np.nonzero(keep)
        return cls(P[rows, cols], cols, indptr, n)

    @classmethod
    def from_csr(cls, data, indices, indptr, n):
        """Adopt an existing CSR triple -- e.g. `organism_compress`'s
        compressed store, which already holds P exactly in this layout, so a
        product that persists compressed never pays the dense scan at all."""
        return cls(data, indices, indptr, n)

    def _rows(self):
        return np.repeat(np.arange(self.n), np.diff(self.indptr))

    @property
    def nnz(self):
        return int(self.data.size)

    @property
    def density(self):
        return self.nnz / max(self.n * self.n, 1)

    def _csc_view(self):
        """In-edges per symbol: Dijkstra relaxes backward from the goal, so
        it needs columns, and a column of a CSR matrix is the one thing CSR
        cannot give cheaply. Built once, reused across goals."""
        if self._csc is None:
            rows = self._rows()
            order = np.argsort(self.indices, kind="stable")
            cols = self.indices[order]
            cindptr = np.zeros(self.n + 1, np.int64)
            np.cumsum(np.bincount(cols, minlength=self.n), out=cindptr[1:])
            self._csc = (rows[order], self._pn[order], cindptr)
        return self._csc

    # ---- the ops ---------------------------------------------------------
    def row(self, i):
        """Row i of the normalized matrix, densified. Same floats as
        `TransitionGraph.normalized(idx)[i]`, zeros included."""
        out = np.zeros(self.n)
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        out[self.indices[lo:hi]] = self._pn[lo:hi]
        return out

    def next_hops(self, goal, eps=1e-12):
        """Bitwise-equal port of `TransitionGraph.next_hops`. Same settle
        order (the argmin scan is kept deliberately -- a heap would reorder
        ties and stop being equal), but relaxation touches only the goal's
        real in-edges instead of a K-long column of mostly zeros, and the
        K^2 logarithms become nnz of them."""
        crows, cvals, cindptr = self._csc_view()
        n = self.n
        dist = np.full(n, np.inf); dist[goal] = 0.0
        nxt = np.full(n, -1, int); nxt[goal] = goal
        done = np.zeros(n, bool)
        for _ in range(n):
            u = int(np.argmin(np.where(done, np.inf, dist)))
            if not np.isfinite(dist[u]) or done[u]:
                break
            done[u] = True
            lo, hi = int(cindptr[u]), int(cindptr[u + 1])
            if hi == lo:
                continue
            v = crows[lo:hi]; q = cvals[lo:hi]
            live = q > eps
            if not live.any():
                continue
            v = v[live]
            cand = dist[u] - np.log(q[live] + eps)
            take = cand < dist[v]
            if take.any():
                nxt[v[take]] = u
                dist[v[take]] = cand[take]
        return nxt, dist

    def plan(self, a, b, eps=1e-12):
        nxt, _ = self.next_hops(b, eps)
        return _trace(nxt, a, b)

    def rollout(self, start, steps, rng):
        """Bitwise-equal port of `TransitionGraph.rollout`, including the RNG
        stream: the row is densified before sampling precisely so that
        `rng.choice` sees the same length and the same probabilities and
        therefore draws the same trajectory. What is saved is the O(K^2)
        normalize the dense op does up front, not the per-step draw."""
        seq = []; cur = int(start)
        for _ in range(steps):
            row = self.row(cur); s = row.sum()
            if s <= 0:
                break
            cur = int(rng.choice(self.n, p=row / s))
            seq.append(cur)
        return np.array(seq, int)

    def rollout_pruned(self, start, steps, rng):
        """Sampling restricted to the real successors -- O(out-degree) per
        step instead of O(K). NOT a port: it consumes the RNG differently, so
        it produces a different (equally valid) trajectory from the same
        seed. Measured separately for exactly that reason; nothing that has
        to reproduce a committed sequence may use it."""
        seq = []; cur = int(start)
        for _ in range(steps):
            lo, hi = int(self.indptr[cur]), int(self.indptr[cur + 1])
            if hi == lo:
                break
            w = self._pn[lo:hi]; s = w.sum()
            if s <= 0:
                break
            cur = int(self.indices[lo + rng.choice(hi - lo, p=w / s)])
            seq.append(cur)
        return np.array(seq, int)

    def matvec_left(self, v):
        """v @ Pn, over nonzeros only."""
        rows = self._rows()
        return np.bincount(self.indices, weights=v[rows] * self._pn,
                           minlength=self.n)

    def kstep_row(self, start, k):
        """Where a symbol lands k hops out -- one row of `kstep`, which is
        the query a planner or a diagnostic actually asks. O(k * nnz) against
        the dense op's O(k * K^3): the dense layer has no way to ask this
        question without building the whole matrix first."""
        v = np.zeros(self.n); v[int(start)] = 1.0
        for _ in range(int(k)):
            v = self.matvec_left(v)
        return v

    def kstep(self, k, block=256):
        """Full k-step matrix, sparse-times-dense in column blocks so the
        nnz x K intermediate never materializes at corpus K."""
        out = np.zeros((self.n, self.n))
        for i in range(self.n):
            lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
            out[i, self.indices[lo:hi]] = self._pn[lo:hi]
        if k == 1:
            return out
        rows = self._rows()
        res = out
        for _ in range(int(k) - 1):
            nxt = np.zeros((self.n, self.n))
            for c0 in range(0, self.n, block):
                B = res[:, c0:c0 + block]
                prod = self._pn[:, None] * B[self.indices]
                seg = np.add.reduceat(prod, self.indptr[:-1], axis=0)
                nz = np.diff(self.indptr) > 0
                nxt[nz, c0:c0 + B.shape[1]] = seg[nz]
            res = nxt
        return res


class SymbolRegistry:
    """T3.3: STABLE SYMBOL IDENTITY, decoupled from slot indices.

    A slot index is storage, not identity. The same word's memory can move
    slots (fusion picks whichever duplicate is the better host), and a slot
    can be reused by an unrelated pattern after recycling/eviction -- so
    `org.P[7]` means different things at different times, and any downstream
    record keyed by slot silently rots. This registry mints an ID the first
    time a slot's content crosses the EventBoundary (i.e. when it becomes a
    symbol the logic layer knows about) and then tracks that ID through every
    identity event the boundary already reports:

      - `mint`       first commit of a slot with no live symbol -> a new ID
      - `fuse`       slot fusion. If the survivor has no ID of its own the
                     dropped slot's ID MOVES to it (identity preserved across
                     a slot change -- the whole point). If both carry IDs the
                     dropped one is ALIASED to the survivor, so an ID handed
                     out earlier still resolves after the merge.
      - `kill`       recycling/eviction. The ID is tombstoned, never reissued,
                     and its slot is freed to mint a fresh ID for whatever
                     pattern lands there next -- a recycled slot must NOT
                     inherit the identity of its previous occupant.

    Every ID in [0, next_id) is therefore in exactly one of three states:
    LIVE (held by a slot), FUSED (aliased onto another ID, resolvable), or
    DEAD (tombstoned). `resolve` chases alias chains with path compression.

    Consolidation is a VIEW, not a mutation: `on_consolidate` records where
    each symbol landed in the compact `org.mem` bank (folded duplicates map
    to their survivor's row) without touching IDs, because callers snapshot
    and restore raw counts around consolidate (see `TransitionGraph.fold`)
    and a destructive registry side-effect there would not be undone by the
    restore. Consolidating twice, or not at all, leaves identity unchanged.

    Purely observational: no mechanism decision reads the registry, so an
    organism with symbols enabled evolves bitwise identically to one without
    (pinned by the harness and `test_fastpath_equivalence.py`). E3 serializes
    it (schema v3)."""

    ALIAS, DEAD = 1, 2                  # journal kinds (numba fastpath drain)

    def __init__(self, K):
        self.K = K
        self.slot_sym = np.full(K, -1, np.int64)   # slot -> live symbol, -1 = none
        self.next_id = np.zeros(1, np.int64)       # array so the JIT can bump it
        self.alias = {}                            # fused-away id -> id it continues as
        self.dead = set()                          # tombstoned ids, never reissued
        self.mem_index = {}                        # symbol -> row of org.mem (a view)

    # ---- boundary notifications (the only mutators) --------------------
    def mint(self, k):
        """Slot k just committed. Give it an ID if it does not have one."""
        if self.slot_sym[k] < 0:
            self.slot_sym[k] = self.next_id[0]
            self.next_id[0] += 1
        return int(self.slot_sym[k])

    def fuse(self, drop, keep):
        """Slot fusion: `drop`'s identity continues as `keep`."""
        sd = int(self.slot_sym[drop])
        if sd >= 0:
            sk = int(self.slot_sym[keep])
            if sk >= 0:
                self.alias[sd] = sk                # both named: drop -> keep
            else:
                self.slot_sym[keep] = sd           # identity moves slots
            self.slot_sym[drop] = -1

    def kill(self, k):
        """Slot k was recycled: its symbol is gone for good."""
        sd = int(self.slot_sym[k])
        if sd >= 0:
            self.dead.add(sd)
            self.slot_sym[k] = -1

    # ---- queries -------------------------------------------------------
    def resolve(self, sid):
        """Follow alias links to the ID this one continues as (itself if it
        was never fused). Path-compressed, so repeated lookups are O(1)."""
        sid = int(sid)
        root = sid
        while root in self.alias:
            root = self.alias[root]
        while sid != root:                          # path compression
            self.alias[sid], sid = root, self.alias[sid]
        return root

    def slot_of(self, sid):
        """Current slot holding this symbol, or -1 if it is dead/unheld.
        This is the migration primitive: index `org.P` by `slot_of(sid)`
        instead of by a slot remembered from an earlier epoch."""
        root = self.resolve(sid)
        hit = np.nonzero(self.slot_sym == root)[0]
        return int(hit[0]) if hit.size else -1

    def symbol_at(self, slot):
        """The symbol currently held by `slot`, or -1."""
        return int(self.slot_sym[slot])

    def status(self, sid):
        """'live' | 'fused' | 'dead' | 'unknown'."""
        sid = int(sid)
        if sid < 0 or sid >= int(self.next_id[0]):
            return 'unknown'
        root = self.resolve(sid)
        if root in self.dead:
            return 'dead'
        if self.slot_of(root) < 0:
            return 'dead'
        return 'live' if root == sid else 'fused'

    def live(self):
        """Every live symbol, ascending."""
        return sorted(int(s) for s in self.slot_sym if s >= 0)

    def minted(self):
        return int(self.next_id[0])

    def mem_row(self, sid):
        """Row of `org.mem` / `org.Pn` this symbol landed in at the last
        consolidate, or -1 (pruned as junk, dead, or never consolidated)."""
        return self.mem_index.get(self.resolve(sid), -1)

    def on_consolidate(self, kept_idx, folds):
        """Rebuild the consolidation VIEW. `kept_idx` is consolidate's
        surviving-slot list (row i of `org.mem` is slot kept_idx[i]);
        `folds` is the (dropped_slot, survivor_slot) pairs it folded."""
        self.mem_index = {}
        for row, slot in enumerate(kept_idx):
            s = int(self.slot_sym[slot])
            if s >= 0:
                self.mem_index[s] = row
        for drop, keep in folds:                    # duplicates share the row
            s, t = int(self.slot_sym[drop]), int(self.slot_sym[keep])
            if s >= 0 and t in self.mem_index:
                self.mem_index[s] = self.mem_index[t]

    # ---- numba fastpath drain -----------------------------------------
    def apply_journal(self, rows):
        """Replay the JIT kernel's identity events. The kernel maintains
        `slot_sym`/`next_id` in place (they are plain arrays) but cannot
        touch the Python alias dict / dead set, so it journals those two
        event kinds in order and they are applied here. Same events, same
        order as the NumPy path -- that equality is what the equivalence
        test pins."""
        for kind, a, b in rows:
            if kind == self.ALIAS:
                self.alias[int(a)] = int(b)
            elif kind == self.DEAD:
                self.dead.add(int(a))


class EventBoundary:
    """BOUNDARY between perception and logic: the single call-site through
    which recognitions become relational knowledge. Perception decides WHEN
    a recognition is committed (confident, non-provisional -- the gate lives
    on the perception side, so ambiguous states never reach the graph); the
    boundary tracks symbol continuity and emits transitions. Identity events
    (fusion, recycling) arrive as explicit notifications so the previous-
    symbol anchor never dangles.

    Those same three notifications are exactly what a stable symbol registry
    needs (T3.3), so when one is attached it is driven from here and nowhere
    else -- no new call sites in the perceive loop, and no way for perception
    to rewrite identity behind the boundary's back."""

    def __init__(self, graph, p_decay=0.0, registry=None):
        self.graph = graph
        self.p_decay = p_decay
        self.registry = registry
        self.prev = -1

    def commit(self, k):
        """A confident, confirmed recognition of symbol k. A change of active
        symbol is a transition; a continued dwell only re-anchors."""
        if self.registry is not None:
            self.registry.mint(k)          # first crossing names the symbol
        if k != self.prev and self.prev >= 0:
            self.graph.observe(self.prev, k, self.p_decay)
        self.prev = k

    def remap(self, drop, keep):
        """Slot fusion: `drop`'s identity continues as `keep`."""
        if self.registry is not None:
            self.registry.fuse(drop, keep)
        if self.prev == drop:
            self.prev = keep

    def invalidate(self, expired):
        """Recycled slots can no longer anchor a transition."""
        if self.registry is not None:
            for k in np.nonzero(expired)[0]:
                self.registry.kill(int(k))
        if self.prev >= 0 and expired[self.prev]:
            self.prev = -1


class Organism:
    def __init__(self, N=128, K=8, omega=0.25, beta=12.0, seed=0, backend=None,
                 symbols=False):
        """backend: 'numba' (require the E2 JIT fastpath), 'numpy' (the
        original interpreted loops), or 'auto' (fastpath when numba is
        importable, numpy otherwise). Default comes from the DEFAI_BACKEND
        env var, falling back to 'auto'. Both backends are pinned by the E1
        regression harness (statistical tolerances -- the JIT legitimately
        reorders float reductions, so bitwise equality is not the bar).

        symbols (T3.3): attach a `SymbolRegistry` giving each committed
        memory an ID that survives fusion, recycling, consolidation and
        save/load. Off by default -- observational only, so an organism with
        it on evolves identically, but the fastpath pays a small per-chunk
        journal for it and every historical caller should stay on the
        untouched path. `org.enable_symbols()` attaches one later."""
        self.backend = backend if backend is not None else os.environ.get("DEFAI_BACKEND", "auto")
        if self.backend not in ("auto", "numba", "numpy"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.backend == "numba" and not fastpath.HAVE_NUMBA:
            raise RuntimeError("backend='numba' requested but numba is not installed")
        self.N, self.K, self.omega, self.beta = N, K, omega, beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.array([normalize(self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N),
                                      self.norm) for _ in range(K)])
        self.used = np.zeros(K, bool)
        self.count = np.zeros(K)            # how often each slot is the confident active memory
        self.age = np.zeros(K)              # slot-budget staleness clock: frames since last
                                            # confident use; persists across perceive calls
                                            # when evict > 0 (T1.2), serialized by E3
        self.evictions = np.zeros(K)        # per-slot pressure-eviction tally (diagnostic)
        self.graph = TransitionGraph(K)     # LOGIC layer: learned transitions between memories
        self.z = normalize(self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N), self.norm)
        self.registry = SymbolRegistry(K) if symbols else None   # T3.3, opt-in

    def enable_symbols(self):
        """Attach a symbol registry to an organism that was built without
        one. Slots already in use are unnamed until they next commit -- IDs
        are minted at the boundary, so identity starts where observation
        starts, not retroactively."""
        if self.registry is None:
            self.registry = SymbolRegistry(self.K)
        return self.registry

    # transition counts live in the logic layer; exposed here (read AND write,
    # slot-indexed) so downstream phase scripts that snapshot/restore or slice
    # org.P keep working unchanged
    @property
    def P(self):
        return self.graph.P

    @P.setter
    def P(self, value):
        self.graph.P = value

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _use_fastpath(self):
        if self.backend == "numpy":
            return False
        if self.backend == "numba":
            return True
        return fastpath.HAVE_NUMBA

    # ---- PHASE 1: perceive + form memories + learn transitions -------------
    def perceive(self, stream, *args, **kw):
        """Compute width is FIXED at complex128 -- see `_perceive` for the
        mechanism itself.

        T1.8 storage compression can leave `xi` narrowed (complex64). The
        numba kernel already promotes its inputs to complex128, so without
        this guard the numpy path would compute the same stream at a
        different width and the two backends would diverge silently. Promote
        the store on entry, restore its dtype on exit; a complex128 store --
        every historical caller -- takes the identity path and is untouched.
        `z` is promoted and left promoted: it is the live field state, O(N)
        bytes, never a compression target, and the fastpath has always
        written it back at compute width."""
        store = self.xi.dtype
        if store == np.complex128 and self.z.dtype == np.complex128:
            return self._perceive(stream, *args, **kw)
        self.xi = np.ascontiguousarray(self.xi, dtype=np.complex128)
        self.z = np.ascontiguousarray(self.z, dtype=np.complex128)
        try:
            return self._perceive(stream, *args, **kw)
        finally:
            self.xi = self.xi.astype(store)

    def _perceive(self, stream, g_in=4.0, dt=0.05, eta=0.02, recruit=0.55, p_decay=0.0,
                  confirm=0, probation=6000, pool=False, active_bar=0.6, s_hat=0.0,
                  amb=0.0, fuse_bar=0.7, evict=0, evict_debug=None):
        """p_decay: exponential forgetting of transition counts, applied once
        per observed transition (synaptic decay). 0.0 = original behavior
        (counts accumulate forever); 1/p_decay is the effective memory in
        transitions. Added in phase 11 to fix concept-drift inertia.

        confirm / probation (phase 14, noise-robust recruitment): confirm > 0
        makes new slots PROVISIONAL -- they must be re-entered on `confirm`
        separate visits (non-contiguous confident matches) within `probation`
        frames to become permanent, else they are recycled. Provisional slots
        are excluded from transition learning, so one-off noise states can
        neither persist as memories nor contaminate P. Rationale: real
        patterns recur, i.i.d. noise does not. confirm=0 = original behavior.
        Biology: synaptic tagging -- traces need reactivation to consolidate.

        pool / active_bar (phase 17, recruitment beyond sigma*): beyond
        sigma* the phase-14 gate fails analytically -- a genuine revisit's
        overlap 1/sqrt(1+sigma^2 N) sits below the hard 0.6 confirmation
        bar, so evidence must be pooled across occurrences BEFORE the
        keep/discard decision. pool=True makes provisional slots EVIDENCE
        POOLS and defers every recruit/keep decision to SETTLED evidence:

          - an input saccade (a jump in the incoming frame) marks the end
            of a token; the field state at that moment -- fully settled on
            the finished token -- is ONE unit of evidence. Mid-settle
            states never recruit: below sigma* they are the junk source
            (phase 14's mid-transition slots), and with the lowered bar
            they would spawn an unmergeable duplicate on nearly every
            token;
          - each slot accepts evidence at an ANNEALED bar matched to
            what its own evidence should look like at its size: 80% of
            the expected same-pattern overlap 1/sqrt((1+s)(1+s/n)) with
            s = s_hat (the noise energy sigma^2 N) and n the slot's
            pooled visits. At n=1 that is the raw token-token overlap
            (a young pool must bootstrap loosely); as the pool denoises
            it tightens toward the memory-grade bar, quenching the
            contamination that loose matching admits. This matters
            because greedy routing against MANY young pools is
            fluctuation-dominated -- the best of ~40 wrong matches
            rivals the one right match, and static bars leave every
            pool a mixture. Evidence goes to the strongest slot that
            clears its own bar. If none does, it recruits a new pool --
            UNLESS a confirmed memory matches it inside that memory's
            dead zone (80% of `active_bar`): such evidence is probably
            the memory's own weak tail, and routing it into a fresh
            pool hands that pool a selection-biased evidence stream.
            (A wider, spread-scaled zone kills duplicates outright but
            locks late words out of recruitment entirely: with many
            confirmed neighbors, the MAX zone fluctuation blocks every
            attempt, and each uncovered word poisons P with skip
            transitions. The narrow zone + slot lifecycle below is the
            working balance.) Young pools cast no dead zone -- shadowing
            novel evidence with half-formed structure starves real
            words of their pools. The match is judged against the slot
            BEFORE the update (held-out: judging a pool against
            evidence it already absorbed is self-fulfilling and lets
            junk confirm itself);
          - accepted evidence updates the slot by running mean with a
            plasticity floor, weight max(1/n, eta): early visits average
            (the trace's noise shrinks ~1/n while the keep/discard
            decision is still open), mature memories keep adapting on an
            ~1/eta-token window. Per-frame refinement is disabled -- at
            high sigma its few-token effective window re-noisifies every
            memory it touches, which is what kept duplicates unmergeable
            and matches flaky below sigma*;
          - a pool graduates when its VISIT QUALITY -- an EMA (1/3) of
            the held-out match quality of its accepted visits -- exceeds
            `active_bar`, with at least `confirm` visits. A young pool's
            per-visit quality is the raw token-token overlap, whose
            distribution has a fat tail: a simple per-visit threshold
            graduates half-denoised duplicates on a few lucky draws.
            Only genuine denoising lifts the SUSTAINED average past the
            bar, so graduation certifies pooled-trace stability: real
            words rise to it, one-off states, mixtures, and evidence-
            starved duplicates never do;
          - slots FUSE online when they converge: noisy assignment
            inevitably splits a word's evidence across duplicate pools,
            but duplicates of the same pattern approach each other as
            they denoise (the plasticity floor re-centers a maturing
            pool on its recent, bar-filtered, nearly-pure evidence, so
            early contamination washes out), while different patterns
            never do -- so when two slots overlap above 0.7 (higher
            than any plausible different-pattern overlap) they merge,
            pooling their evidence, counts, and transitions. Splitting
            is thereby harmless instead of fatal: the halves find each
            other;
          - USE IT OR LOSE IT: any slot -- provisional or confirmed --
            that goes a full probation window without accepting a
            single visit is recycled, and its transition rows cleared.
            This is the lifecycle answer to the one duplicate that
            survives everything else: a contaminated pool that
            graduates and then FREEZES (its annealed bar rises past
            what its mixed content can attract, so it accepts nothing,
            forever) while still soaking up activity counts. A real
            memory keeps resonating with the stream that made it; dead
            structure fades and frees its slot.

        amb (phase 18, mixture hygiene at the source): phase 17's residual
        was frozen MIXTURE slots born in bootstrap routing chaos -- with
        many young pools at loose bars, greedy routing is fluctuation-
        dominated, and no per-slot statistic can undo the mixing after
        the fact (five cures tried, all failed; see phase 17). The gate
        acts BEFORE the mixing instead, and it is SOFT: while the winning
        pool is still provisional, its evidence is weighted by an
        attribution CONFIDENCE conf = clip(margin/amb, 0, 1), where
        margin is the winner's lead over the runner-up across all live
        slots. Confidence scales the whole absorption -- evidence mass
        (nvis += conf), the running-mean step (weight max(conf/n,
        conf*eta)), the visit-quality EMA, and staying alive
        (age *= 1-conf instead of age = 0, so a pool that subsists on
        contested scraps expires while one with regular confident wins
        stays young). Rationale: assignment errors are precisely the
        low-margin events -- a wrong best-of-many RIVALS the right
        match, it rarely dominates it -- so mixing events barely move
        any pool, while a word's ~150 recurrences arrive mostly at
        decent margins. Mature (confirmed) winners absorb at full
        weight: their annealed bar already rejects wrong-word tokens,
        and down-weighting them only starves adaptation. At sigma=0
        margins are huge (same-word ~1.0 vs cross ~0.15), so conf
        saturates at 1 and the clean case is untouched by construction.
        amb=0.0 reproduces phase-17 behavior exactly.

        Variants tried and rejected (measured at sigma=0.3, phase 18):
        a HARD margin gate (drop contested tokens) re-traces the
        coverage-for-purity collapse of phase 17's wide dead zones
        (coverage 1.00 -> 0.69 by amb=0.15); routing deeply ambiguous
        tokens to RECRUITMENT fragments evidence into endless fresh
        pools (coverage 0.27 at rec=0.1); removing the weak-tail dead
        zone lets orphaned words recruit again but their pools then
        subsist on low-confidence evidence and never graduate. The soft
        gate with the dead zone kept is the only variant that moved the
        purity-coverage frontier outward instead of sliding along it.

        fuse_bar (phase 26, percentile calibration): the duplicate-fusion
        threshold, previously the hardcoded constant 0.7. Exposed so it can
        be placed from the measured similarity distribution instead of
        assumed; 0.7 default reproduces prior behavior exactly.

        evict (T1.2, slot budget / eviction under recruitment pressure):
        phase 33 measured the failure this closes -- in a class-incremental
        stream the first task's data floods every slot (recruit is a
        similarity floor, so recruitment is eager while slots are free),
        after which `not used.all()` blocks recruitment FOREVER and later
        patterns can only be learned by dragging existing slots off their
        memories (slot drift: a slow forgetting channel, the FORG 0.20).
        evict > 0 turns the pool-mode use-it-or-lose-it recycling into a
        DEMAND-DRIVEN budget, in any mode:

          - the staleness clock self.age (frames since the slot was last
            confidently used) becomes PERSISTENT organism state -- it no
            longer resets at perceive-call boundaries, because capacity
            pressure is a property of the organism's whole life, not of
            one stream. In plain/confirm modes a slot's clock zeroes on
            confident activity (the same event that feeds `count`); in
            pool mode the clock stays acceptance-driven, exactly the
            existing use-it-or-lose-it semantics (a frozen mixture that
            soaks up activity while accepting nothing must still read as
            dead). E3 serializes the clock; old state files load with a
            zeroed clock.
          - eviction fires only UNDER RECRUITMENT PRESSURE: a novel token
            wants a slot and none is free. Then the least-established
            stale slot -- argmin lifetime `count` among slots idle longer
            than `evict` frames, the same rarely-the-confident-active-
            memory junk signal consolidate() prunes by -- is recycled
            through the existing machinery (used/count cleared,
            graph.retire, boundary.invalidate), and recruitment proceeds
            into the freed slot. Nothing is evicted while free slots
            remain, and slots idle less than `evict` frames are never
            touched, so an active stream cannot churn live memories.
            Count-ordering makes the budget reclaim junk and duplicates
            before established memories; timer-expiry (probation) is
            unchanged and still owns the pool-mode lifecycle.

        Label-free by construction: staleness and usage counts only --
        no task boundaries, no class information anywhere. evict=0 (the
        default) reproduces prior behavior exactly, including leaving
        self.age untouched.

        evict_debug (T1.7, diagnostic only): when a list is passed, every
        pressure eviction appends one row (frame, slot, count, age) -- the
        victim's lifetime count and staleness age AT eviction, with `frame`
        the 0-based index into this call's stream. Pure observation: rows
        are recorded before the eviction mutates anything and no mechanism
        decision reads them, so evict_debug=None (the default) and an
        attached ledger produce bitwise-identical organism state (pinned by
        phase 33e's A/B check). Era/birth attribution is deliberately NOT
        recorded here -- the mechanism is label- and task-blind; ledger
        consumers reconstruct it (phase 33e replays rebirths script-side).

        active_bar is also the confidence bar for counting and transition
        learning (0.6 = original); beyond sigma* it must sit below the
        token-vs-memory overlap or P never learns. pool=False with
        active_bar=0.6 reproduces phase-14 behavior exactly."""
        if self._use_fastpath():
            return fastpath.perceive_fast(
                self, stream, g_in=g_in, dt=dt, eta=eta, recruit=recruit,
                p_decay=p_decay, confirm=confirm, probation=probation,
                pool=pool, active_bar=active_bar, s_hat=s_hat, amb=amb,
                fuse_bar=fuse_bar, evict=evict, evict_debug=evict_debug)
        z = self.z
        boundary = EventBoundary(self.graph, p_decay, self.registry)
        prov = np.zeros(self.K, bool)
        hits = np.zeros(self.K); nvis = np.zeros(self.K)
        # with evict > 0 the staleness clock is the persistent self.age (the
        # budget outlives any one stream); otherwise the historical per-call
        # local, so evict=0 behavior is untouched
        age = self.age if evict > 0 else np.zeros(self.K)
        prev_x = None
        last_k = -1

        def evict_one():
            """Slot budget under recruitment pressure: reclaim the least-
            established stale slot (argmin lifetime count among slots idle
            > evict frames). Returns True if a slot was freed."""
            ev = self.used & (age > evict)
            if not ev.any():
                return False
            j = int(np.argmin(np.where(ev, self.count, np.inf)))
            if evict_debug is not None:            # T1.7 ledger: observe-only,
                evict_debug.append((frame_t, j,    # recorded before mutation
                                    float(self.count[j]), float(age[j])))
            self.used[j] = False; self.count[j] = 0
            prov[j] = False; hits[j] = 0; nvis[j] = 0; age[j] = 0
            self.evictions[j] += 1
            self.graph.retire(j)
            expired = np.zeros(self.K, bool); expired[j] = True
            boundary.invalidate(expired)
            return True
        for frame_t, x in enumerate(stream):
            x = normalize(x, self.norm)
            if pool and confirm > 0:
                if prev_x is not None and np.linalg.norm(x - prev_x) > 0.5 * self.norm:
                    # saccade: z is the finished token's settled state -- the
                    # unit of evidence; all recruit/pool decisions live here
                    mm = np.abs(self.overlaps(z, self.xi))
                    mm[~self.used] = -1.0                  # random init is not a match
                    bars = 0.8 / np.sqrt((1 + s_hat) * (1 + s_hat / np.maximum(nvis, 1.0)))
                    cand = mm > bars
                    conf = 1.0
                    if amb > 0 and cand.any():             # ambiguity gate (phase 18)
                        kk = int(np.argmax(np.where(cand, mm, -1.0)))
                        if prov[kk]:                       # mature winners absorb freely
                            rest = np.where(self.used, mm, -1.0)
                            rest[kk] = -1.0
                            conf = min(1.0, max(0.0, (mm[kk] - rest.max(initial=-1.0)) / amb))
                    if cand.any() and conf > 0.0:          # strongest accepting slot wins
                        kk = int(np.argmax(np.where(cand, mm, -1.0)))
                        o_s = (self.xi[kk].conj() @ z) / self.N
                        z_al = z * np.exp(-1j * np.angle(o_s))
                        nvis[kk] += conf                   # accepting evidence = staying alive,
                        age[kk] *= (1.0 - conf)            # in proportion to its confidence
                        wv = max(conf / nvis[kk], eta * conf)  # running mean, plasticity floor
                        self.xi[kk] = normalize(
                            self.xi[kk] + wv * (z_al - self.xi[kk]), self.norm)
                        if prov[kk]:                       # visit-quality EMA: sustained
                            hits[kk] += conf * (mm[kk] - hits[kk]) / 3.0  # confidence, not lucky draws
                            if hits[kk] > active_bar and nvis[kk] > confirm:
                                prov[kk] = False           # graduated: stable pooled trace
                        oo = np.abs(self.overlaps(self.xi[kk], self.xi))
                        oo[kk] = 0.0; oo[~self.used] = 0.0
                        j = int(np.argmax(oo))
                        if oo[j] > fuse_bar:               # converged duplicates: fuse
                            keep, drop = (kk, j) if (prov[j], nvis[kk]) > (prov[kk], nvis[j]) else (j, kk)
                            w_d = nvis[drop] / (nvis[keep] + nvis[drop])
                            o_kd = (self.xi[keep].conj() @ self.xi[drop]) / self.N
                            al = self.xi[drop] * np.exp(-1j * np.angle(o_kd))
                            self.xi[keep] = normalize(
                                self.xi[keep] + w_d * (al - self.xi[keep]), self.norm)
                            nvis[keep] += nvis[drop]
                            hits[keep] = max(hits[keep], hits[drop])
                            prov[keep] = prov[keep] and prov[drop]
                            self.count[keep] += self.count[drop]
                            self.graph.merge(keep, drop)
                            self.used[drop] = False; prov[drop] = False
                            self.count[drop] = 0
                            boundary.remap(drop, keep)
                    elif mm[self.used & ~prov].max(initial=0.0) < 0.8 * active_bar:
                        # novel (not a memory's weak tail)
                        if evict > 0 and self.used.all():
                            evict_one()                    # budget: reclaim stale structure
                        if not self.used.all():
                            f = int(np.argmin(self.used.astype(float)))
                            self.xi[f] = normalize(z, self.norm); self.used[f] = True
                            prov[f] = True; hits[f] = 0; age[f] = 0; nvis[f] = 1
                prev_x = x
            dz = 1j * self.omega * z + g_in * (x - z)     # input-driven (no retrieval: avoids collapse)
            z = normalize(z + dt * dz, self.norm)
            o = self.overlaps(z, self.xi); m = np.abs(o)
            k = int(np.argmax(m))
            if pool and confirm > 0:
                pass                                       # all updates happen at saccades
            else:
                if evict > 0 and m[k] < 0.8 * recruit and self.used.all():
                    # budget: reclaim stale structure. Recruiting out of a
                    # FREED slot demands more novelty (0.8x, the pool-mode
                    # weak-tail fraction) than recruiting into a free one:
                    # near-misses refine an existing memory instead of
                    # spending the budget, so a pattern stops evicting once
                    # it holds prototypes at that spacing. Without this
                    # throttle, demand at recruit-spacing flushes the whole
                    # bank every regime change (measured, phase 33b).
                    evict_one()
                if m[k] < recruit and not self.used.all():  # novel -> recruit a free slot
                    f = int(np.argmin(self.used.astype(float)))
                    self.xi[f] = normalize(z, self.norm); self.used[f] = True; k = f
                    age[f] = 0
                    if confirm > 0:
                        prov[f] = True; hits[f] = 0
                else:                                      # familiar -> refine memory
                    z_al = z * np.exp(-1j * np.angle(o[k]))
                    self.xi[k] = normalize(self.xi[k] + eta * (z_al - self.xi[k]), self.norm)
                    self.used[k] = True
            if confirm > 0:
                if not pool and prov[k] and m[k] > 0.6 and k != last_k:  # a fresh visit, not the same dwell
                    hits[k] += 1
                    if hits[k] >= confirm:
                        prov[k] = False                     # graduated: pattern recurs
                if pool:
                    # use it or lose it: ANY slot that stops accepting
                    # evidence -- an unconfirmed one-off, or a frozen
                    # mixture whose annealed bar outgrew its mixed
                    # content -- is dead structure and is recycled
                    age[self.used] += 1
                    expired = self.used & (age > probation)
                else:
                    if evict > 0:
                        age[self.used] += 1   # budget clock: every used slot ages
                    else:                     # (prov in used, so provisional birth-
                        age[prov] += 1        # clocks tick identically either way)
                    expired = prov & (age > probation)
                if expired.any():                           # recycle
                    self.used[expired] = False; self.count[expired] = 0
                    prov[expired] = False
                    if pool:
                        self.graph.retire(expired)
                        boundary.invalidate(expired)
            elif evict > 0:                                 # plain mode: budget clock only
                age[self.used] += 1
            if m[k] > active_bar and self.used[k]:
                self.count[k] += 1
                if not prov[k]:                            # gate: only confirmed, confident
                    if evict > 0 and not pool:             # confident use resets the budget
                        age[k] = 0                         # clock (pool: acceptance owns it)
                    boundary.commit(k)                     # recognitions cross into logic
            last_k = k
        if confirm > 0:                                    # purge still-unconfirmed slots
            if self.registry is not None:
                # a provisional slot normally has no symbol (minting happens
                # at commit, which is gated on NOT provisional) -- except in
                # pool mode, where fusion can move a confirmed slot's
                # identity onto a still-provisional host. Purging that host
                # kills the symbol like any other recycling.
                boundary.invalidate(prov)
            self.used[prov] = False; self.count[prov] = 0
        self.z = z

    # ---- consolidate: merge duplicate memories, drop unused ---------------
    def consolidate(self, merge_thresh=0.8, prune_frac=0.05):
        # prune junk slots (rarely the confident active memory -> recruited mid-transition)
        keepable = self.count > prune_frac * self.count.max()
        idx = list(np.where(self.used & keepable)[0])
        # all pairwise overlaps in one matmul; the greedy first-duplicate scan
        # below then touches only scalars (the per-pair Python-level dot made
        # this O(K^2) in interpreter dispatch at corpus scale)
        # compute width is complex128 even when the STORE is narrowed
        # (T1.8), so consolidation products are backend-identical
        xi = np.ascontiguousarray(self.xi, dtype=complex)
        Xk = xi[idx] if idx else xi[:0]
        O = np.abs(Xk @ Xk.conj().T) / self.N
        merged = []; merged_pos = []; folds = []
        for pk, k in enumerate(idx):
            dup = next((j for pj, j in zip(merged_pos, merged)
                        if O[pk, pj] > merge_thresh), None)
            if dup is None:
                merged.append(k); merged_pos.append(pk)
            else:                                          # fold transitions into the kept slot
                self.graph.fold(dup, k)
                folds.append((k, dup))
        if self.registry is not None:
            # a VIEW over the compact bank, not an identity mutation: callers
            # snapshot/restore raw counts around consolidate (see graph.fold),
            # so IDs must come out the far side unchanged
            self.registry.on_consolidate(merged, folds)
        self.mem = xi[merged]
        self.kept_idx = merged   # indices into self.P/self.xi that survived -- lets
                                 # callers reconstruct RAW (unnormalized) transition
                                 # counts restricted to the kept memories later.
        self.Pn = self.graph.normalized(merged)            # row-normalized transition probs
        return merged

    # ---- PHASE 13: recall with lateral inhibition + hop commitment ---------
    def recall2(self, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
                g_rec=5.0, Dn=0.004, topk=8, commit=0.6, debounce=20):
        """Two fixes over recall(), each ablatable:
        - topk: only the k best-scoring memories compete for the field pull
          (lateral inhibition). With many crowded memories, the full softmax
          blends same-category attractors into a blob and the state flickers
          among them; top-k keeps the pull discrete. topk >= K disables.
        - commit/debounce: a transition is recorded only when the new memory
          exceeds `commit` overlap AND stays the argmax for `debounce`
          consecutive steps -- mid-flight states no longer count as hops.
          debounce=1, commit=0.5 reproduces recall()'s acceptance rule."""
        if self._use_fastpath():
            return fastpath.recall2_fast(self, steps=steps, dt=dt, tau_h=tau_h,
                                         lam=lam, gamma=gamma, g_rec=g_rec, Dn=Dn,
                                         topk=topk, commit=commit, debounce=debounce)
        M = self.mem; Ku = M.shape[0]
        z = normalize(self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N), self.norm)
        h = np.zeros(Ku); seq = []; cur = 0
        cand = -1; streak = 0
        k_eff = min(topk, Ku)
        for s in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat = np.maximum(1 - lam * h, 0.0)
            score = m * fat + gamma * self.Pn[cur] * fat
            if k_eff < Ku:
                top = np.argpartition(score, -k_eff)[-k_eff:]
                w = np.zeros(Ku)
                e = np.exp(self.beta * (score[top] - score[top].max()))
                w[top] = e / e.sum()
            else:
                w = np.exp(self.beta * (score - score.max())); w /= w.sum()
            phase = o / (m + 1e-9)
            T = (w * phase) @ M
            noise = np.sqrt(2 * Dn * dt) * (self.rng.standard_normal(self.N) +
                                            1j * self.rng.standard_normal(self.N)) / np.sqrt(2)
            z = normalize(z + dt * (1j * self.omega * z + g_rec * (T - z)) + noise, self.norm)
            h = h + dt / tau_h * (m - h)
            a = int(np.argmax(m))
            if a != cur and m[a] > commit:
                streak = streak + 1 if a == cand else 1
                cand = a
                if streak >= debounce:
                    seq.append(a); cur = a; streak = 0
            else:
                streak = 0; cand = -1
        return np.array(seq)

    # ---- PHASE 30: goal-directed recall -- logic proposes, field renders ---
    def recall_directed(self, goal, steps=40000, dt=0.05, tau_h=18.0, lam=2.0,
                        gamma=2.5, g_rec=5.0, Dn=0.004):
        """Reasoning driving generation across the boundary: the LOGIC layer
        plans the most-probable route to `goal` (a kept-memory index) on the
        learned graph, and at each moment proposes only the next hop; the
        field's job reduces to rendering and settling on the proposal. Same
        dynamics as recall() -- habituation, softmax pull, noise -- with the
        undirected prior Pn[cur] replaced by the plan's one-hot next hop.
        Stops when the goal is committed. Returns (seq, reached)."""
        M = self.mem; Ku = M.shape[0]
        z = normalize(self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N), self.norm)
        h = np.zeros(Ku); seq = []; cur = 0
        nxt, _ = self.graph.next_hops(self.kept_idx, goal)
        for s in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat = np.maximum(1 - lam * h, 0.0)
            prior = np.zeros(Ku)
            if nxt[cur] >= 0:
                prior[nxt[cur]] = 1.0       # the plan's proposal for this moment
            score = m * fat + gamma * prior * fat
            w = np.exp(self.beta * (score - score.max())); w /= w.sum()
            phase = o / (m + 1e-9)
            T = (w * phase) @ M
            noise = np.sqrt(2 * Dn * dt) * (self.rng.standard_normal(self.N) +
                                            1j * self.rng.standard_normal(self.N)) / np.sqrt(2)
            z = normalize(z + dt * (1j * self.omega * z + g_rec * (T - z)) + noise, self.norm)
            h = h + dt / tau_h * (m - h)
            a = int(np.argmax(m))
            if m[a] > 0.5:
                if a != cur:
                    seq.append(a)
                cur = a
                if cur == goal:
                    return np.array(seq), True
        return np.array(seq), False

    # ---- PHASE 2: recall = generate learned trajectories ------------------
    def recall(self, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
               g_rec=5.0, Dn=0.004):
        if self._use_fastpath():
            return fastpath.recall_fast(self, steps=steps, dt=dt, tau_h=tau_h,
                                        lam=lam, gamma=gamma, g_rec=g_rec, Dn=Dn)
        M = self.mem; Ku = M.shape[0]
        z = normalize(self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N), self.norm)
        h = np.zeros(Ku); seq = []; cur = 0
        for s in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat = np.maximum(1 - lam * h, 0.0)
            # selection = freshness (fatigue) + learned transition prior from current
            score = m * fat + gamma * self.Pn[cur] * fat
            w = np.exp(self.beta * (score - score.max())); w /= w.sum()
            phase = o / (m + 1e-9)
            T = (w * phase) @ M
            noise = np.sqrt(2 * Dn * dt) * (self.rng.standard_normal(self.N) +
                                            1j * self.rng.standard_normal(self.N)) / np.sqrt(2)
            z = normalize(z + dt * (1j * self.omega * z + g_rec * (T - z)) + noise, self.norm)
            h = h + dt / tau_h * (m - h)
            a = int(np.argmax(m))
            if m[a] > 0.5:
                if a != cur:
                    seq.append(a)
                cur = a
        return np.array(seq)


# ============================== DEMO / EVAL ================================
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, H, K = 128, 4, 8
    NORM = np.sqrt(N)

    # hidden world: H regimes + a STRUCTURED transition graph (mostly a cycle
    # 0->1->2->3->0 with a little branching). Unknown to the organism.
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    Ttrue = np.array([[0.0, 0.8, 0.1, 0.1],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.1, 0.1, 0.0, 0.8],
                      [0.8, 0.1, 0.1, 0.0]])
    def make_stream(n, dwell=60, noise=0.5):
        h = 0; out = []
        for i in range(n):
            if i % dwell == 0 and i > 0:
                h = rng.choice(H, p=Ttrue[h])
            out.append(G[h] + noise * NORM / np.sqrt(N) *
                       (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
        return out

    org = Organism(N=N, K=K, seed=0)
    org.perceive(make_stream(80000))
    kept = org.consolidate()

    print("ORGANISM (Phase 1): memories + learned world structure + recall\n")
    print(f"slots used {org.used.sum()}/{K} -> after consolidate: {len(kept)} memories")

    # (A) memory capture
    cap = [max(np.abs(org.overlaps(G[h], org.mem))) for h in range(H)]
    print(f"\n(A) regime capture (never saw clean patterns): mean {np.mean(cap):.3f}  {[f'{c:.2f}' for c in cap]}")

    # map each kept memory -> true regime
    mem2reg = [int(np.argmax(np.abs(org.overlaps(org.mem[k], G)))) for k in range(org.mem.shape[0])]
    print(f"    memory->regime map: {mem2reg}")

    # (B) learned transition structure vs truth
    print("\n(B) learned transition matrix (rows=from, cols=to), in REGIME order:")
    Preg = np.zeros((H, H))
    for i in range(org.mem.shape[0]):
        for j in range(org.mem.shape[0]):
            Preg[mem2reg[i], mem2reg[j]] += org.Pn[i, j]
    Preg /= (Preg.sum(1, keepdims=True) + 1e-9)
    for h in range(H):
        print(f"    from {h}: learned {np.round(Preg[h],2)}   true {Ttrue[h]}")

    # (C) recall generates sequences matching learned structure (vs random baseline)
    seq = org.recall(60000)
    reg_seq = np.array([mem2reg[s] for s in seq])
    # empirical bigram of recalled regime sequence
    B = np.zeros((H, H))
    for a, b in zip(reg_seq[:-1], reg_seq[1:]):
        if a != b: B[a, b] += 1
    Bn = B / (B.sum(1, keepdims=True) + 1e-9)
    # score: correlation of recalled transitions with true; baseline = uniform
    mask = ~np.eye(H, dtype=bool)
    corr = np.corrcoef(Bn[mask], Ttrue[mask])[0, 1]
    # baseline: shuffle the recalled sequence -> destroys order, keeps regime freqs
    shuf = reg_seq.copy(); rng.shuffle(shuf)
    Bs = np.zeros((H, H))
    for a, b in zip(shuf[:-1], shuf[1:]):
        if a != b: Bs[a, b] += 1
    Bsn = Bs / (Bs.sum(1, keepdims=True) + 1e-9)
    base = np.corrcoef(Bsn[mask], Ttrue[mask])[0, 1]
    print(f"\n(C) recall ({len(seq)} transitions). recalled-vs-true transition corr = {corr:.3f}")
    print(f"    (shuffled-sequence baseline corr = {base:.3f})")
    print(f"    recalled regime sequence (first 40): {reg_seq[:40].tolist()}")

    ok = np.mean(cap) > 0.7 and corr > 0.5
    print("\nverdict:", "ORGANISM learns world STRUCTURE and recalls plausible trajectories"
          if ok else "partial -- inspect stage")
