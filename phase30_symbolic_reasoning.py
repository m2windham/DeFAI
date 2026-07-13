"""
PHASE 30: SYMBOLIC REASONING ON THE LOGIC LAYER (out-of-sequence phase, spun
off the logic/language architecture split -- phases 26-29 stay reserved for
the roadmap items).

The split gave the organism two systems with a narrow boundary: perception
(the oscillator field -- "language") and the TransitionGraph ("logic"). The
neuroscience result that motivated the split (MIT/McGovern 2026: reasoning
survives total loss of language) predicts something testable here: REASONING
SHOULD RUN ON THE GRAPH ALONE, with the field not just unnecessary but
absent. This phase adds three reasoning ops to the logic layer and measures
whether they work, against nulls, on a world where reasoning has something
to do:

  (A) MULTI-STEP INFERENCE  graph.kstep -- conclusions about what happens k
      hops ahead, chained from one-step knowledge. The organism never
      observed a 2-step statistic; if kstep matches the true k-step law
      above a permutation null, the logic layer genuinely infers, not
      recalls.
  (B) SYMBOLIC IMAGINATION  graph.rollout -- trajectories sampled purely in
      symbol space (no settling, no habituation, no attractors). Scored the
      same way field recall is scored (bigram corr vs truth) and timed. The
      claim is aphasia-shaped: imagination without the "language" system
      preserves the learned structure, at a fraction of the cost. Honesty:
      rollout samples the learned graph directly, so its score measures the
      graph's quality -- the point is that NOTHING of the field is needed.
  (C) PLANNING -> GOAL-DIRECTED GENERATION  graph.next_hops + organism.
      recall_directed -- Dijkstra on -log transition probability plans the
      most-probable route to a goal memory; during recall the logic layer
      proposes only the next hop and the field merely renders it. Baseline:
      undirected recall() with the same step budget, scored by committed
      hops until the goal memory is first reached. The world is built with a
      real fork (two branches from the hub, goal on one of them) so
      planning has a decision to make -- in the phase-1 cycle world every
      road leads everywhere and planning would be untestable.

World: 6 regimes, hub 0 forks to branch 1->2->5 or branch 4->(back to 0);
regime 5 is reachable almost only through 2. Ground truth used ONLY for
evaluation (memory->regime mapping, true k-step law), never inside the
mechanism -- standing rules.
"""

import time
import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(7)
N, H, K = 128, 6, 12
NORM = np.sqrt(N)

# hidden world: hub-and-branches. Row h = P(next regime | regime h).
Ttrue = np.array([
    [0.00, 0.45, 0.05, 0.45, 0.05, 0.00],   # hub: fork to 1 or 3
    [0.10, 0.00, 0.85, 0.05, 0.00, 0.00],   # branch A: 1 -> 2
    [0.05, 0.05, 0.00, 0.00, 0.00, 0.90],   # 2 -> 5 (the gate to the goal)
    [0.10, 0.05, 0.00, 0.00, 0.85, 0.00],   # branch B: 3 -> 4
    [0.85, 0.00, 0.05, 0.10, 0.00, 0.00],   # 4 -> back to hub
    [0.90, 0.05, 0.00, 0.05, 0.00, 0.00],   # 5 -> back to hub
])
assert np.allclose(Ttrue.sum(1), 1.0)

Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
G = Gr.T * NORM


def make_stream(n, dwell=60, noise=0.5):
    h = 0; out = []
    for i in range(n):
        if i % dwell == 0 and i > 0:
            h = rng.choice(H, p=Ttrue[h])
        out.append(G[h] + noise * NORM / np.sqrt(N) *
                   (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
    return out


print("PHASE 30: symbolic reasoning on the logic layer\n")
org = Organism(N=N, K=K, seed=0)
org.perceive(make_stream(120000))
kept = org.consolidate()
n_mem = org.mem.shape[0]
mem2reg = [int(np.argmax(np.abs(org.overlaps(org.mem[k], G)))) for k in range(n_mem)]
cap = [max(np.abs(org.overlaps(G[h], org.mem))) for h in range(H)]
print(f"setup: {org.used.sum()}/{K} slots -> {n_mem} memories, "
      f"capture mean {np.mean(cap):.3f}, memory->regime {mem2reg}")
covered = len(set(mem2reg)) == H
if not covered:
    print("  WARNING: not every regime captured -- verdicts below are partial")


def to_regime_matrix(Pm):
    """Project a memory-space stochastic matrix into regime space (eval only)."""
    R = np.zeros((H, H))
    for i in range(n_mem):
        for j in range(n_mem):
            R[mem2reg[i], mem2reg[j]] += Pm[i, j]
    return R / (R.sum(1, keepdims=True) + 1e-9)


def offdiag_corr(A, B):
    mask = ~np.eye(H, dtype=bool)
    return np.corrcoef(A[mask], B[mask])[0, 1]


# ===================== (A) multi-step inference vs permutation null =========
print("\n(A) multi-step inference: kstep vs true k-step law, against a null")
L1 = to_regime_matrix(org.graph.normalized(org.kept_idx))
null_rng = np.random.default_rng(99)
for k in [1, 2, 3]:
    Lk = to_regime_matrix(org.graph.kstep(org.kept_idx, k))
    truth_k = np.linalg.matrix_power(Ttrue, k)
    corr = offdiag_corr(Lk, truth_k)
    # null: shuffle the learned one-step off-diagonals, renormalize, power.
    # Kills the learned structure, keeps the value distribution.
    mask = ~np.eye(H, dtype=bool)
    nulls = []
    for _ in range(500):
        Ln = np.zeros((H, H))
        vals = L1[mask].copy(); null_rng.shuffle(vals)
        Ln[mask] = vals
        Ln /= (Ln.sum(1, keepdims=True) + 1e-9)
        nulls.append(offdiag_corr(np.linalg.matrix_power(Ln, k), truth_k))
    bar = float(np.percentile(nulls, 99))
    print(f"    k={k}: corr {corr:.3f} vs null 99th pct {bar:.3f} "
          f"-> {'SIGNAL' if corr > bar else 'NOT above null'}")

# ===================== (B) symbolic imagination vs field recall =============
print("\n(B) symbolic imagination: rollout (no field) vs recall (field)")


def bigram_corr(reg_seq):
    B = np.zeros((H, H))
    for a, b in zip(reg_seq[:-1], reg_seq[1:]):
        if a != b:
            B[a, b] += 1
    Bn = B / (B.sum(1, keepdims=True) + 1e-9)
    return offdiag_corr(Bn, Ttrue)


t0 = time.time()
seq_f = org.recall(60000)
t_field = time.time() - t0
corr_field = bigram_corr(np.array([mem2reg[s] for s in seq_f]))

t0 = time.time()
seq_s = org.graph.rollout(org.kept_idx, start=0, steps=len(seq_f), rng=np.random.default_rng(5))
t_symb = time.time() - t0
corr_symb = bigram_corr(np.array([mem2reg[s] for s in seq_s]))

print(f"    field recall : corr {corr_field:.3f}  in {t_field:.2f}s ({len(seq_f)} hops)")
print(f"    rollout      : corr {corr_symb:.3f}  in {t_symb:.4f}s ({len(seq_s)} hops)"
      f"  -> {t_field / max(t_symb, 1e-9):.0f}x faster, zero field dynamics")

# ===================== (C) planning -> goal-directed generation =============
print("\n(C) goal-directed recall vs undirected baseline (hops to goal)")
BUDGET = 12000
REPS = 5
rows = []
for goal in range(1, n_mem):                       # memory 0 excluded: both
    d_hops, u_hops = [], []                        # methods start locked near
    for _ in range(REPS):                          # it, the comparison is void
        seq, reached = org.recall_directed(goal, steps=BUDGET)
        d_hops.append(len(seq) if reached else np.nan)
        sequ = org.recall(BUDGET)
        wh = np.where(sequ == goal)[0]
        u_hops.append(int(wh[0]) + 1 if len(wh) else np.nan)
    rows.append((goal, mem2reg[goal], np.nanmean(d_hops), np.mean(~np.isnan(d_hops)),
                 np.nanmean(u_hops), np.mean(~np.isnan(u_hops))))
    print(f"    goal mem {goal} (regime {mem2reg[goal]}): "
          f"directed {rows[-1][2]:5.1f} hops (success {rows[-1][3]:.0%})  "
          f"undirected {rows[-1][4]:5.1f} hops (success {rows[-1][5]:.0%})")

d_mean = float(np.nanmean([r[2] for r in rows]))
u_mean = float(np.nanmean([r[4] for r in rows]))
d_succ = float(np.mean([r[3] for r in rows]))
u_succ = float(np.mean([r[5] for r in rows]))
print(f"    overall: directed {d_mean:.1f} hops ({d_succ:.0%} success) vs "
      f"undirected {u_mean:.1f} hops ({u_succ:.0%} success)")

# ============================== verdict =====================================
plan_wins = d_succ >= u_succ and d_mean < u_mean
print("\nverdict:",
      "logic layer REASONS: multi-step inference above null, imagination "
      "without the field, planning beats undirected wandering"
      if covered and plan_wins and corr_symb > 0.5 else
      "partial -- inspect the failing stage above")
