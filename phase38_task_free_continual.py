"""
PHASE 38 (T6.1): TASK-FREE CONTINUAL LEARNING -- does the retention advantage
hold on a stream that never announces when the lesson changes?

WHY THIS AND NOT ANOTHER LADDER RUN. The split-digits ladder (phase 33/33c)
hands every learner explicit task boundaries. That is the one thing the
organism does not need and every gradient arm does: EWC needs a boundary to
snapshot Fisher information, replay needs one to balance its buffer. Scoring
the organism on a boundaried protocol measures a BORROWED condition. This
phase measures the differentiated one, which is also the deployment condition
for an episodic-memory product: streams do not announce their tasks.

THE DESIGN ISOLATES THE VARIABLE. Three conditions over identical pacing
(20 chunks, identical optimizer-step budget, identical organism frame count),
so the only things that move are the stream's composition and whether the
gradient arms are told where the transitions are:

  A  BOUNDARIED    hard task switches; gradient arms TOLD (EWC snapshots at
                   true task ends, replay buffer balanced per task).
                   The internal control. Phase 33c is the external anchor.
  B  HARD-UNTOLD   the SAME stream as A, gradient arms NOT told: EWC snapshots
                   on a fixed period-5 schedule (chunks 4/9/14/19) that is
                   plausible but misaligned with the true transitions
                   (chunks 3/7/11/15/19); replay switches to reservoir
                   sampling. Isolates the SIGNAL from the stream.
  C  TASK-FREE     gradual class drift + interleaved re-emergence of earlier
                   classes + unannounced novelty, arms not told. The full
                   deployment condition, and the one T6.1 names.

The organism arms are BIT-FOR-BIT THE SAME CODE in all three conditions. They
have no boundary input to remove -- that is the entire point, and it is why
the A->C delta is the measurement rather than any single condition's number.

Labels enter only at the eval-side readout (`LabelEvidenceReadout`, T1.3) and
in the supervised baselines, exactly as in 33c. The eval class-pairs are used
to SCORE, never to route: no arm receives the task identity of a sample.

PRE-REGISTERED PREDICTIONS (committed before the run that tests them; the
falsification branch is named and its decision rule is fixed here):

 (a) THE GAP NARROWS FROM THE OTHER SIDE. Going A -> C, the boundary-dependent
     gradient arms (mlp-ewc, mlp-replay) lose more accuracy than ORGANISM+B.
     Decision rule: dACC(A->C) for BOTH mlp-ewc and mlp-replay is more
     negative than dACC(A->C) for ORGANISM+B, on >= 4 of 5 seeds.
     The framing is deliberately honest: this is the gap closing WITHOUT the
     organism improving. The clause is only claimed if the organism's own
     dACC(A->C) >= -0.02 (i.e. it did not itself degrade materially). If the
     organism improves by more than +0.02 the clause is recorded as
     CONFOUNDED rather than confirmed -- an easier stream is not a result.

 (b) RECRUITMENT LOCALIZES THE TRANSITIONS. In condition C, the organism's own
     fresh-slot recruitment rate is higher in the chunks where a class first
     appears (unannounced novelty) than elsewhere, above a permutation null.
     Decision rule: z >= 2.0 against 2000 permutations of the novelty-chunk
     labels, on the seed-pooled statistic, with the per-seed sign consistent
     on >= 4 of 5 seeds. This is the axis the gradient arms need to be TOLD:
     if it holds, the organism computes for itself the signal EWC is handed.

 (c) HONEST NEGATIVE, and it is a real possibility. If ORGANISM+B's
     dACC(A->C) <= -0.10, the organism degrades sharply too, task-free
     operation is NOT a free differentiator, and the claim must be dropped
     rather than softened. Decision rule is exactly that threshold. A partial
     outcome -- organism degrades but less than the gradient arms -- is
     reported as such and named a WEAKER claim, not a win.

WHAT THIS PHASE CANNOT SHOW, stated before the numbers. This is one dataset
(sklearn digits, 10 classes, 64 features) at one capacity (K=40). It is a
class-incremental image protocol, not language, and the organism's headline
differentiators (unsupervised representation, structure learning, bitwise
persistence) are not what ACC measures. A dACC comparison is a statement
about THIS protocol's sensitivity to boundary information, not a general
claim about continual learning. Phase 33c's verdict is untouched by this
phase: on raw class-incremental accuracy with boundaries, the organism is
not SOTA and this phase does not revisit that.

NO GRID IS SEARCHED. Every constant is inherited from a committed phase --
K=40, evict=250, hold=8, omega/beta/g_in/eta/recruit from 33c verbatim; the
optimizer-step budget is matched to 33c's 5x300. Nothing here is selected on
the measured outcome, so the held-out-seed rule (T1.6/T1.8/T1.9) is not
triggered. All arms are reseeded s=0-4 and the baselines are reseeded on the
SAME per-seed stream as the organism -- an unpaired fixed-seed baseline is a
bug (T1.9), so every arm in a given seed sees the identical chunk sequence.
"""

import time

import numpy as np

try:                    # baselines only -- organism arms run without torch
    import torch
    import torch.nn as nn
except ImportError:
    torch = nn = None
from sklearn.datasets import load_digits

from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_numba import NumbaOrganism

# ---- data pipeline: identical to phase 33/33c, including the split draw ----
_d = load_digits()
X = ((_d.data - _d.data.mean(0)) / (_d.data.std(0) + 1e-6)).astype(np.float32)
y = _d.target.astype(int)
N = X.shape[1]
NORM = np.sqrt(N)
_perm = np.random.default_rng(0).permutation(len(X))
tr, te = _perm[:1400], _perm[1400:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
T = len(TASKS)

# ---- protocol constants, all inherited (see docstring: no grid) ----
K = 40                  # 33c
EVICT = 250             # 33b's primary window, via 33c
HOLD = 8                # 33c
EPOCHS_EQUIV = 3        # 33c's organism epochs -> total stream draws
CHUNKS = 20             # pacing granularity: 4 per task-equivalent
CHUNK_N = EPOCHS_EQUIV * len(Xtr) // CHUNKS      # 210 draws per chunk
STEPS_PER_CHUNK = 75    # 20 x 75 = 1500 = 33c's 5 x 300 optimizer steps
JOINT_STEPS = 300       # 33c's joint oracle, unchanged
EVAL_EVERY = CHUNKS // T                          # 5 checkpoints, as in 33c
REEMERGE = 0.15         # condition C: fraction drawn from already-seen tasks
UNTOLD_PERIOD = 5       # misaligned fixed snapshot schedule for B and C
SEEDS = range(5)
N_PERM = 2000           # prediction (b) permutation null

# pre-registered decision thresholds (see docstring)
PRED_A_MIN_SEEDS = 4
PRED_A_ORG_FLOOR = -0.02
PRED_B_Z = 2.0
PRED_C_COLLAPSE = -0.10


def task_acc_matrix_to_metrics(A):
    """A[i, j] = accuracy on task j's classes at checkpoint i. 33c's formula."""
    final = A[T - 1]
    acc = float(np.mean(final))
    forg = float(np.mean([np.max(A[:, j]) - final[j] for j in range(T - 1)]))
    return acc, forg


def eval_tasks(predict):
    out = []
    for task in TASKS:
        m = np.isin(yte, task)
        out.append(float((predict(Xte[m]) == yte[m]).mean()))
    return out


class _Pool:
    """Per-task index queue that reshuffles when exhausted, so draws are
    without replacement within a pass. Identical machinery in every
    condition -- the conditions differ only in HOW MANY draws each task
    contributes to a chunk."""

    def __init__(self, idx, rng):
        self.idx, self.rng, self.pos = np.asarray(idx), rng, 0
        self.order = rng.permutation(len(idx))

    def take(self, n):
        out = []
        while n > 0:
            if self.pos >= len(self.order):
                self.order, self.pos = self.rng.permutation(len(self.idx)), 0
            k = min(n, len(self.order) - self.pos)
            out.append(self.idx[self.order[self.pos:self.pos + k]])
            self.pos += k
            n -= k
        return np.concatenate(out)


def build_stream(condition, seed):
    """Return (chunks, mix) where chunks[c] is the array of train indices for
    chunk c and mix[c][t] is the number of draws task t contributed.

    A/B: chunk c is pure task c // EVAL_EVERY (hard switches).
    C  : composition ramps between consecutive tasks, plus REEMERGE of the
         already-introduced tasks -- gradual drift, interleaved re-emergence,
         and unannounced novelty in one stream.
    """
    rng = np.random.default_rng(1000 + seed)
    pools = [_Pool(np.where(np.isin(ytr, t))[0], rng) for t in TASKS]
    chunks, mix = [], []
    for c in range(CHUNKS):
        counts = np.zeros(T, dtype=int)
        if condition in ('A', 'B'):
            counts[c // EVAL_EVERY] = CHUNK_N
        else:
            u = (c + 0.5) * T / CHUNKS          # continuous position in tasks
            tb = min(int(u), T - 1)
            f = u - tb                          # ramp weight onto tb + 1
            n_re = int(round(REEMERGE * CHUNK_N)) if tb > 0 else 0
            n_main = CHUNK_N - n_re
            if tb + 1 < T:
                counts[tb] += int(round((1 - f) * n_main))
                counts[tb + 1] += n_main - int(round((1 - f) * n_main))
            else:
                counts[tb] += n_main
            for t in rng.integers(0, tb + 1, size=n_re):   # already-seen only
                counts[t] += 1
        take = [pools[t].take(int(counts[t])) for t in range(T)]
        ch = np.concatenate([a for a in take if len(a)])
        chunks.append(ch[rng.permutation(len(ch))])
        mix.append(counts)
    return chunks, np.array(mix)


def novelty_chunks(mix):
    """Chunks in which a task's classes appear for the FIRST time -- the
    unannounced-novelty moments prediction (b) is about."""
    seen, out = set(), []
    for c, counts in enumerate(mix):
        for t in range(T):
            if counts[t] > 0 and t not in seen:
                seen.add(t)
                out.append(c)
    return sorted(set(out))


# ------------------------------- gradient arms -------------------------------
if torch is not None:
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N, 128), nn.ReLU(),
                                     nn.Linear(128, 10))

        def forward(self, x):
            return self.net(x)


def mlp_predict(mlp):
    def f(Xe):
        with torch.no_grad():
            return mlp(torch.tensor(Xe)).argmax(1).numpy()
    return f


def run_mlp(mode, chunks, mix, condition, seed, ewc_lam=1000.0,
            buf_size=200):
    """33c's gradient recipes, driven by the chunk stream.

    told = condition A. EWC snapshots Fisher at true task ends and replay
    keeps a per-task balanced buffer. Untold (B and C) replaces both with the
    boundary-free versions every task-free paper has to use: a fixed-period
    snapshot schedule and reservoir sampling.
    """
    torch.manual_seed(seed)
    mlp, told = MLP(), condition == 'A'
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    A = np.zeros((T, T))
    fishers, stars = [], []
    buf_i, seen_n = [], 0
    rng = np.random.default_rng(7000 + seed)
    t0 = time.time()

    if mode == 'joint':                       # offline oracle, condition-free
        xb, yb = torch.tensor(Xtr), torch.tensor(ytr)
        for _ in range(JOINT_STEPS):
            opt.zero_grad(); lossf(mlp(xb), yb).backward(); opt.step()
        A[:] = eval_tasks(mlp_predict(mlp))
        return A, time.time() - t0

    for c, ch in enumerate(chunks):
        cur = ch
        if mode == 'replay' and buf_i:
            cur = np.concatenate([ch, np.array(buf_i, dtype=int)])
        xb, yb = torch.tensor(Xtr[cur]), torch.tensor(ytr[cur])
        for _ in range(STEPS_PER_CHUNK):
            opt.zero_grad()
            loss = lossf(mlp(xb), yb)
            if mode == 'ewc':
                for F, st in zip(fishers, stars):
                    for p, f_, s_ in zip(mlp.parameters(), F, st):
                        loss = loss + (ewc_lam / 2) * (f_ * (p - s_) ** 2).sum()
            loss.backward(); opt.step()

        if mode == 'ewc':
            snap = ((c % EVAL_EVERY) == EVAL_EVERY - 1) if told \
                else ((c % UNTOLD_PERIOD) == UNTOLD_PERIOD - 1)
            if snap:
                F = [torch.zeros_like(p) for p in mlp.parameters()]
                for i in range(0, len(ch), 8):
                    mlp.zero_grad()
                    out = mlp(torch.tensor(Xtr[ch[i:i + 8]]))
                    nn.functional.nll_loss(
                        nn.functional.log_softmax(out, 1),
                        torch.tensor(ytr[ch[i:i + 8]])).backward()
                    for f_, p in zip(F, mlp.parameters()):
                        f_ += p.grad.detach() ** 2
                fishers.append([f_ / max(len(ch) // 8, 1) for f_ in F])
                stars.append([p.detach().clone() for p in mlp.parameters()])

        if mode == 'replay':
            if told:                          # per-task balanced, 33c's rule
                per = buf_size // T
                keep = []
                for t in range(T):
                    ti = [i for i in list(buf_i) + list(ch)
                          if ytr[i] in TASKS[t]]
                    keep.extend(ti[:per])
                buf_i = keep
            else:                             # reservoir over the raw stream
                for i in ch:
                    seen_n += 1
                    if len(buf_i) < buf_size:
                        buf_i.append(int(i))
                    else:
                        j = int(rng.integers(0, seen_n))
                        if j < buf_size:
                            buf_i[j] = int(i)

        if (c % EVAL_EVERY) == EVAL_EVERY - 1:
            A[c // EVAL_EVERY] = eval_tasks(mlp_predict(mlp))
    return A, time.time() - t0


def run_prototypes(chunks, thresh=0.55):
    """33c's supervised memory-method bar, driven by the same stream. Needs no
    boundary, so it is identical in every condition -- a useful control on
    whether a condition is simply harder."""
    protos, labels = [], []
    A = np.zeros((T, T))
    t0 = time.time()
    for c, ch in enumerate(chunks):
        for i in ch:
            v = Xtr[i] / (np.linalg.norm(Xtr[i]) + 1e-9)
            if protos:
                sims = np.array([v @ p for p in protos])
                j = int(np.argmax(sims))
                if sims[j] > thresh:
                    protos[j] = protos[j] + 0.1 * (v - protos[j])
                    protos[j] /= np.linalg.norm(protos[j]) + 1e-9
                    continue
            protos.append(v); labels.append(ytr[i])
        if (c % EVAL_EVERY) == EVAL_EVERY - 1:
            def pred(Xe, P=np.array(protos), L=np.array(labels)):
                Xn = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9)
                return L[np.argmax(Xn @ P.T, axis=1)]
            A[c // EVAL_EVERY] = eval_tasks(pred)
    return A, time.time() - t0, len(protos)


def run_organism(chunks, evict, seed, cls=NumbaOrganism):
    """33c's organism arm, driven by the chunk stream. IDENTICAL in every
    condition: no boundary is passed in, because there is none to pass. The
    readout follows slot identity under eviction (33b's rule, 33c verbatim).
    Returns the per-chunk fresh-recruit counts as well -- that series is
    prediction (b)'s measurement."""
    org = cls(N=N, K=K, omega=0.15, beta=10.0, seed=seed)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    A = np.zeros((T, T))
    recruits = []
    t0 = time.time()
    for c, ch in enumerate(chunks):
        used_before, evd_before = org.used.copy(), org.evictions.copy()
        seq = []
        for i in ch:
            seq.extend([normalize(Xtr[i].astype(complex), NORM)] * HOLD)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict)
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        recruits.append(int(fresh.sum()))
        readout.invalidate(fresh)
        readout.observe(org, Xtr[ch], ytr[ch])
        if (c % EVAL_EVERY) == EVAL_EVERY - 1:
            A[c // EVAL_EVERY] = eval_tasks(
                lambda Xe: readout.predict(org, Xe))
    return A, time.time() - t0, np.array(recruits), int(org.used.sum())


def recruit_localization(recruits, nov, rng):
    """Prediction (b): is recruitment concentrated in the novelty chunks?
    Statistic is the difference of means (novelty vs rest); the null permutes
    WHICH chunks carry the novelty label, holding the recruitment series
    fixed, so it tests localization rather than overall level."""
    recruits = np.asarray(recruits, dtype=float)
    m = np.zeros(len(recruits), dtype=bool)
    m[nov] = True
    if m.all() or not m.any():
        return np.nan, np.nan, np.nan
    obs = recruits[m].mean() - recruits[~m].mean()
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        p = rng.permutation(len(recruits))
        null[b] = recruits[p][m].mean() - recruits[p][~m].mean()
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return float(obs), float(z), float((null >= obs).mean())


if __name__ == "__main__":
    print("PHASE 38 (T6.1): task-free continual learning -- boundary-free stream\n")
    if torch is None:
        print("  torch not installed -- the gradient arms ARE the comparison")
        print("  in prediction (a). A run without them is not the deliverable.\n")

    conds = ['A', 'B', 'C']
    label = {'A': 'BOUNDARIED', 'B': 'HARD-UNTOLD', 'C': 'TASK-FREE'}
    res = {c: {} for c in conds}
    recs = {}
    joint = {}

    for seed in SEEDS:
        print(f"seed {seed}")
        if torch is not None:                 # offline oracle: condition-free
            Aj, _ = run_mlp('joint', None, None, 'A', seed)
            joint[seed] = task_acc_matrix_to_metrics(Aj)
        for cond in conds:
            chunks, mix = build_stream(cond, seed)
            nov = novelty_chunks(mix)
            arms = {}
            if torch is not None:
                for m in ('seq', 'ewc', 'replay'):
                    A, t = run_mlp(m, chunks, mix, cond, seed)
                    arms[f'mlp-{m}'] = task_acc_matrix_to_metrics(A)
            A, t, npr = run_prototypes(chunks)
            arms['prototypes'] = task_acc_matrix_to_metrics(A)
            A, t, r0, u0 = run_organism(chunks, 0, seed)
            arms['organism'] = task_acc_matrix_to_metrics(A)
            A, t, r1, u1 = run_organism(chunks, EVICT, seed)
            arms['ORGANISM+B'] = task_acc_matrix_to_metrics(A)
            res[cond][seed] = arms
            if cond == 'C':
                recs[seed] = (r1, nov)
            print(f"  {label[cond]:<12} " + "  ".join(
                f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in arms.items()))

    names = list(res['A'][0].keys())
    print("\n" + "=" * 70)
    print("ACC / FORG, mean over 5 paired seeds (same stream per seed)")
    print(f"  {'arm':<12} " + "  ".join(f"{label[c]:>18}" for c in conds))
    for nme in names:
        cells = []
        for c in conds:
            a = np.mean([res[c][s][nme][0] for s in SEEDS])
            f = np.mean([res[c][s][nme][1] for s in SEEDS])
            cells.append(f"{a:.3f}/{f:.3f}".rjust(18))
        print(f"  {nme:<12} " + "  ".join(cells))
    if joint:
        print(f"  {'mlp-joint*':<12} " +
              f"{np.mean([joint[s][0] for s in SEEDS]):.3f} (offline oracle, "
              "condition-free)")

    print("\ndACC(A->C), per arm -- the measurement (negative = degraded):")
    d = {}
    for nme in names:
        per = [res['C'][s][nme][0] - res['A'][s][nme][0] for s in SEEDS]
        d[nme] = per
        print(f"  {nme:<12} {np.mean(per):+.3f}   per-seed " +
              " ".join(f"{v:+.3f}" for v in per))

    print("\n" + "=" * 70)
    print("PRE-REGISTERED VERDICTS")

    org_d = np.mean(d['ORGANISM+B'])
    if 'mlp-ewc' in d:
        wins = sum(1 for s in range(len(list(SEEDS)))
                   if d['mlp-ewc'][s] < d['ORGANISM+B'][s]
                   and d['mlp-replay'][s] < d['ORGANISM+B'][s])
        ok_a = wins >= PRED_A_MIN_SEEDS and org_d >= PRED_A_ORG_FLOOR
        state = 'CONFIRMED' if ok_a else 'NOT CONFIRMED'
        if org_d > 0.02:
            state = 'CONFOUNDED (organism improved -- easier stream)'
        print(f"  (a) gap narrows from the other side: {state}")
        print(f"      both gradient arms degrade more on {wins}/5 seeds "
              f"(rule: >= {PRED_A_MIN_SEEDS}); organism dACC {org_d:+.3f} "
              f"(floor {PRED_A_ORG_FLOOR:+.3f})")
    else:
        print("  (a) NOT TESTABLE -- torch absent, gradient arms did not run")

    pooled_r = np.concatenate([recs[s][0] for s in SEEDS])
    pooled_n = np.concatenate([np.array(recs[s][1]) + i * CHUNKS
                               for i, s in enumerate(SEEDS)])
    obs, z, p = recruit_localization(pooled_r, pooled_n,
                                     np.random.default_rng(99))
    signs = [recruit_localization(recs[s][0], recs[s][1],
                                  np.random.default_rng(200 + s))[0] > 0
             for s in SEEDS]
    ok_b = (z >= PRED_B_Z) and sum(signs) >= PRED_A_MIN_SEEDS
    print(f"  (b) recruitment localizes novelty: "
          f"{'CONFIRMED' if ok_b else 'NOT CONFIRMED'}")
    print(f"      pooled diff {obs:+.2f} slots/chunk, z {z:.2f} "
          f"(rule: >= {PRED_B_Z}), perm p {p:.4f}, "
          f"per-seed sign positive {sum(signs)}/5")

    if org_d <= PRED_C_COLLAPSE:
        print(f"  (c) HONEST NEGATIVE FIRED: organism dACC {org_d:+.3f} "
              f"<= {PRED_C_COLLAPSE:+.3f} -- task-free operation is NOT a free")
        print("      differentiator on this protocol; the claim must be "
              "dropped, not softened.")
    else:
        print(f"  (c) collapse clause did not fire (organism dACC "
              f"{org_d:+.3f} > {PRED_C_COLLAPSE:+.3f})")

    print("\nscope: one dataset (digits), one capacity (K=40), image "
          "classification.\n  dACC measures THIS protocol's sensitivity to "
          "boundary information.\n  Phase 33c's boundaried verdict is "
          "untouched: the organism is not SOTA there.")
