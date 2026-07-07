"""
PHASE 14 -- NOISE-ROBUST PERCEPTION: probationary recruitment

Phase 13 fixed recall but left the noise case honestly open: at sigma=0.3
per-token corruption the organism stored 42 memories for 26 words, and no
recall policy can fix what memory contains. Diagnosis of the storage side:

  - At sigma=0.3 per dimension on unit embeddings in 30-d, the noise energy
    is ~2.7x the signal energy per token. Settled states can land below the
    0.5 recruit overlap against every stored memory -- and the gate recruits.
  - The junk slot then persists forever: it soaks up counts, enters the
    Hebbian transition matrix, and survives the 2% prune.

The asymmetry that fixes it: REAL PATTERNS RECUR, I.I.D. NOISE DOES NOT.
A word appears ~150 times in a 4000-word stream; a noise-displaced state is
never seen again. So make recruitment evidence-gated
(Organism.perceive(confirm=..., probation=...)):

  - a newly recruited slot is PROVISIONAL;
  - it graduates only after `confirm` separate visits (non-contiguous
    confident matches) within `probation` frames; else it is recycled;
  - provisional slots are excluded from transition learning, so P stays
    clean even while candidates are on probation.

confirm=0 reproduces the original gate exactly. Biology: synaptic
tagging-and-capture -- a trace must be reactivated to consolidate.

Protocol (small world, 4000 words, recall2 generation throughout):
  - sigma in {0.0, 0.3, 0.6} x confirm in {0 (baseline), 2, 3}
  - plus the blunt alternative at sigma=0.3: confirm=0 with a 10% prune --
    is simple pruning enough, or does the junk have to be kept out of P
    while learning?
Success criteria: at sigma=0.3, slots ~= vocabulary and generation well
above phase 13's 0.466; the clean case must be UNHARMED (a hygiene
mechanism that taxes normal learning is not accepted).
"""

import numpy as np
from organism import Organism, normalize

ANIMAL, ACTION, OBJECT = 0, 1, 2
NEXT_FWD = {0: 1, 1: 2, 2: 0}
P_CORRECT = 0.88
HOLD = 12

ANIMALS = ['cat','dog','bird','horse','cow','pig','sheep','wolf']
ACTIONS = ['run','jump','swim','eat','sleep','hunt','hide','play']
OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
vocab = ANIMALS + ACTIONS + OBJECTS
V = len(vocab)
cats = np.array([ANIMAL]*8 + [ACTION]*8 + [OBJECT]*10)

N = 30; NORM = np.sqrt(N)
emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, N))
cat_bases[0, 0:3] = 1.0; cat_bases[1, 3:6] = 1.0; cat_bases[2, 6:9] = 1.0
emb = np.zeros((V, N))
for i in range(V):
    emb[i] = 0.6*cat_bases[cats[i]] + 0.4*emb_rng.standard_normal(N)
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9


def sample_stream(n, seed):
    local = np.random.default_rng(seed)
    pools = [np.where(cats == c)[0] for c in range(3)]
    cat = ANIMAL
    seq = []
    for _ in range(n):
        seq.append(int(local.choice(pools[cat])))
        cat = NEXT_FWD[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != NEXT_FWD[cat]]))
    return seq


def frames(seq, noise, seed=1):
    local = np.random.default_rng(seed)
    out = []
    for w in seq:
        e = emb[w] + (noise * local.standard_normal(N) if noise > 0 else 0)
        out.extend([normalize(e.astype(complex), NORM)] * HOLD)
    return out


def evaluate(org):
    """Coverage, junk share of memories, prediction, recall2 generation."""
    states = np.array([normalize(emb[w].astype(complex), NORM) for w in range(V)])
    used_idx = np.where(org.used)[0]
    n_slots_raw = len(used_idx)
    ovr = np.abs((org.xi[used_idx].conj() @ states.T) / N)
    w2s_raw = {w: int(used_idx[ovr[:, w].argmax()]) for w in range(V)}
    s2w_raw = {int(used_idx[i]): int(ovr[i].argmax()) for i in range(n_slots_raw)}
    coverage = np.mean([s2w_raw[w2s_raw[w]] == w for w in range(V)])
    # prediction from raw P (before consolidation compacts it)
    probe = sample_stream(1000, seed=7)
    ok = tot = 0
    for a, b in zip(probe[:-1], probe[1:]):
        row = org.P[w2s_raw[a]]
        if row.sum() <= 0: continue
        j = int(np.argmax(row))
        if j in s2w_raw:
            ok += cats[s2w_raw[j]] == cats[b]; tot += 1
    pred = ok / max(tot, 1)
    # generation with recall2 (the adopted default)
    org.consolidate(merge_thresh=0.84, prune_frac=0.02)
    M = org.mem
    ovc = np.abs((M.conj() @ states.T) / N)
    s2w = {i: int(ovc[i].argmax()) for i in range(M.shape[0])}
    dedicated = set(int(ovc[:, w].argmax()) for w in range(V))
    seq = [int(s) for s in org.recall2(steps=60000, topk=8, debounce=20)][:400]
    junk_hops = np.mean([s not in dedicated for s in seq]) if seq else 0.0
    gen = [s2w[s] for s in seq]
    gram = (np.mean([NEXT_FWD[cats[a]] == cats[b] for a, b in zip(gen[:-1], gen[1:])])
            if len(gen) > 1 else 0.0)
    return dict(slots=n_slots_raw, cov=coverage, pred=pred,
                gram=gram, junk=junk_hops, hops=len(seq))


def run(sigma, confirm, prune_frac=None, label=None):
    org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    seq = sample_stream(4000, seed=99)
    org.perceive(frames(seq, sigma), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5,
                 confirm=confirm)
    if prune_frac is not None:   # blunt-fix comparison arm
        org.consolidate(merge_thresh=0.84, prune_frac=prune_frac)
    r = evaluate(org)
    print(f"  {label or f'confirm={confirm}':<22} slots={r['slots']:>3}  "
          f"coverage={r['cov']:.2f}  prediction={r['pred']:.3f}  "
          f"generation={r['gram']:.3f}  junk hops={r['junk']:.3f}  hops={r['hops']}")
    return r


results = {}
for sigma in (0.0, 0.3, 0.6):
    print(f"\n=== sigma={sigma} (26 words, 4000-word stream, recall2 generation) ===")
    for confirm in (0, 2, 3):
        results[(sigma, confirm)] = run(sigma, confirm)
print("\n=== blunt alternative at sigma=0.3: no gating, 10% prune ===")
results['blunt'] = run(0.3, 0, prune_frac=0.10, label="confirm=0 + prune 10%")

print("\n" + "="*70)
print("PHASE 14 SUMMARY -- generation grammaticality (chance ~0.33)")
print(f"{'sigma':>6} {'confirm=0':>10} {'confirm=2':>10} {'confirm=3':>10}")
for sigma in (0.0, 0.3, 0.6):
    print(f"{sigma:>6} " + " ".join(f"{results[(sigma, c)]['gram']:>10.3f}" for c in (0, 2, 3)))
print(f"\nblunt 10%-prune arm at sigma=0.3: {results['blunt']['gram']:.3f} "
      f"({results['blunt']['slots']} slots pre-prune)")

clean_ok = results[(0.0, 2)]['gram'] >= results[(0.0, 0)]['gram'] - 0.03
best_c = max((2, 3), key=lambda c: results[(0.3, c)]['gram'])
gain = results[(0.3, best_c)]['gram'] - results[(0.3, 0)]['gram']
if clean_ok and gain > 0.15:
    print(f"\nverdict: probationary recruitment WORKS -- confirm={best_c} lifts sigma=0.3 "
          f"generation {results[(0.3,0)]['gram']:.2f} -> {results[(0.3,best_c)]['gram']:.2f} "
          "with the clean case unharmed; adopt for noisy deployments")
elif clean_ok:
    print("\nverdict: hygiene is clean-safe but the noise gain is modest -- "
          "junk storage was not the (only) binding constraint; re-diagnose")
else:
    print("\nverdict: gating taxes clean learning -- not accepted as-is; "
          "tune probation/confirm or re-diagnose")
