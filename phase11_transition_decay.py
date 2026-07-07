"""
PHASE 11 -- TRANSITION DECAY: graceful forgetting fixes concept-drift inertia

The deployment benchmark (simulate_scenarios.py, scenario C) measured the
architecture's worst deployment property: when the world's grammar reversed
mid-stream, the organism adapted from 0.12 to only 0.44 on the new regime
after 4000 words of evidence. Cause: Hebbian transition counts accumulate
forever, so old evidence outvotes new in proportion to its sheer volume --
the more the organism has lived, the slower it changes its mind.

Fix (one line, organism-native): exponential decay of the transition matrix,
applied once per observed transition (synaptic decay / recency weighting).
`Organism.perceive(p_decay=...)`; 1/p_decay is the effective memory length
in transitions. p_decay=0 reproduces the original behavior exactly.

This is also the first Tier-4 property from the roadmap: a lifelong learner
must forget at the rate its world changes.

Measured questions:
  1. ADAPTATION: same drift protocol as scenario C (4x1000 words forward
     grammar, then 4x1000 reversed). How fast does next-category prediction
     on the NEW regime recover, per decay rate?
  2. STABILITY COST: in a stationary world (4000 words, no drift), does
     decay hurt steady-state generation? (It shouldn't: the normalized
     transition structure is unchanged in expectation; only the effective
     sample size shrinks.)
  3. RECOMMENDATION: the smallest decay that fixes adaptation, with its
     measured stationary cost stated -- not hidden.

RESULT (recorded from the committed run):
  - p_decay=0.001 lifts new-regime prediction from 0.44 to 0.87 (~the 0.88
    oracle ceiling), and recovery is nearly immediate: 0.73 within the first
    1000 words after the reversal, vs 0.16 without decay.
  - Cost: stationary prediction drops 0.82 -> 0.75 (effective sample size
    shrinks); generation is unaffected (0.63 vs 0.62). Old-regime accuracy
    after drift falls to ~0.09 -- with decay the organism COMMITS to the
    current world instead of hedging badly on both (0.48/0.44 without).
  - RECOMMENDED: p_decay=0.001 for any non-stationary deployment; keep 0.0
    only for provably stationary worlds where the 7pp prediction edge
    matters more than adaptivity.
"""

import numpy as np
from organism import Organism, normalize

# ---- world (identical to simulate_scenarios.py) --------------------------------
ANIMALS = ['cat','dog','bird','horse','cow','pig','sheep','wolf']
ACTIONS = ['run','jump','swim','eat','sleep','hunt','hide','play']
OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
vocab = ANIMALS + ACTIONS + OBJECTS
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
ANIMAL, ACTION, OBJECT = 0, 1, 2
CAT = {}
for w in ANIMALS: CAT[word_to_idx[w]] = ANIMAL
for w in ACTIONS: CAT[word_to_idx[w]] = ACTION
for w in OBJECTS: CAT[word_to_idx[w]] = OBJECT
NEXT_FWD = {0: 1, 1: 2, 2: 0}
NEXT_REV = {0: 2, 2: 1, 1: 0}

DIM = 30; N = DIM; NORM = np.sqrt(N)
P_CORRECT = 0.88
HOLD = 12

emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, DIM))
cat_bases[0, 0:3] = 1.0; cat_bases[1, 3:6] = 1.0; cat_bases[2, 6:9] = 1.0
embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(vocab):
    embeddings[i] = 0.6*cat_bases[CAT[i]] + 0.4*emb_rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

FULL = {ANIMAL: ANIMALS, ACTION: ACTIONS, OBJECT: OBJECTS}


def sample_stream(n, next_cat, seed):
    local = np.random.default_rng(seed)
    cat = ANIMAL
    seq = []
    for _ in range(n):
        seq.append(word_to_idx[local.choice(FULL[cat])])
        cat = next_cat[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != next_cat[cat]]))
    return seq


def frames(seq):
    out = []
    for w in seq:
        s = normalize(embeddings[w].astype(complex), NORM)
        out.extend([s] * HOLD)
    return out


def word_slot_maps(org):
    M = org.xi[org.used]
    used_idx = np.where(org.used)[0]
    states = np.array([normalize(embeddings[w].astype(complex), NORM) for w in range(N_WORDS)])
    ov = np.abs((M.conj() @ states.T) / N)
    w2s = {w: int(used_idx[np.argmax(ov[:, w])]) for w in range(N_WORDS)}
    s2w = {int(used_idx[i]): int(np.argmax(ov[i])) for i in range(len(used_idx))}
    return w2s, s2w


def predict_accuracy(org, seq_eval):
    w2s, s2w = word_slot_maps(org)
    ok = tot = 0
    for a, b in zip(seq_eval[:-1], seq_eval[1:]):
        row = org.P[w2s[a]]
        if row.sum() <= 0: continue
        j = int(np.argmax(row))
        if j in s2w:
            ok += CAT[s2w[j]] == CAT[b]; tot += 1
    return ok / max(tot, 1)


def recall_grammaticality(org, next_cat, n_gen=400, steps=60000):
    org.consolidate(merge_thresh=0.84, prune_frac=0.02)
    M = org.mem
    states = np.array([normalize(embeddings[w].astype(complex), NORM) for w in range(N_WORDS)])
    ov = np.abs((M.conj() @ states.T) / N)
    s2w = {i: int(np.argmax(ov[i])) for i in range(M.shape[0])}
    seq = org.recall(steps=steps)
    gen = [s2w[int(s)] for s in seq][:n_gen]
    ok = sum(1 for a, b in zip(gen[:-1], gen[1:]) if next_cat[CAT[a]] == CAT[b])
    return ok / max(len(gen)-1, 1)


probe_fwd = sample_stream(800, NEXT_FWD, seed=7)
probe_rev = sample_stream(800, NEXT_REV, seed=8)
chunks = [sample_stream(1000, NEXT_FWD, seed=100+i) for i in range(4)] + \
         [sample_stream(1000, NEXT_REV, seed=200+i) for i in range(4)]

DECAYS = [0.0, 0.001, 0.003, 0.01]

print("="*70)
print("1. ADAPTATION under drift (grammar reverses after chunk 4)")
print("   value shown: next-category prediction accuracy on the NEW regime (REV)")
header = f"{'after chunk':>12}" + "".join(f"  decay={d:<7}" for d in DECAYS)
print(header)
adapt = {d: [] for d in DECAYS}
fwd_end = {}
orgs = {d: Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0) for d in DECAYS}
for i, ch in enumerate(chunks):
    row = f"{i+1:>12}"
    for d in DECAYS:
        orgs[d].perceive(frames(ch), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, p_decay=d)
        acc = predict_accuracy(orgs[d], probe_rev)
        adapt[d].append(acc)
        row += f"  {acc:<13.3f}"
    print(row + ("   <-- drift" if i == 3 else ""))
for d in DECAYS:
    fwd_end[d] = predict_accuracy(orgs[d], probe_fwd)
print("\n  old-regime (FWD) accuracy after all 8 chunks: "
      + "  ".join(f"decay={d}: {fwd_end[d]:.3f}" for d in DECAYS))

print("\n" + "="*70)
print("2. STABILITY COST in a stationary world (4000 words, no drift)")
stationary = {}
for d in DECAYS:
    org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
    org.perceive(frames(sample_stream(4000, NEXT_FWD, seed=99)),
                 g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, p_decay=d)
    acc = predict_accuracy(org, probe_fwd)
    gram = recall_grammaticality(org, NEXT_FWD)
    stationary[d] = (acc, gram)
    print(f"  decay={d:<6} prediction={acc:.3f}  generation grammaticality={gram:.3f}")

print("\n" + "="*70)
print("PHASE 11 SUMMARY")
print(f"{'decay':>7} {'REV acc @end':>13} {'stationary pred':>16} {'stationary gen':>15}")
for d in DECAYS:
    print(f"{d:>7} {adapt[d][-1]:>13.3f} {stationary[d][0]:>16.3f} {stationary[d][1]:>15.3f}")

base_rev = adapt[0.0][-1]
# acceptance: adaptation must improve substantially; stationary prediction may
# pay up to 10pp and generation up to 5pp (the trade-off is stated, not hidden)
candidates = [d for d in DECAYS[1:]
              if stationary[d][0] >= stationary[0.0][0] - 0.10
              and stationary[d][1] >= stationary[0.0][1] - 0.05
              and adapt[d][-1] > base_rev + 0.1]
if candidates:
    rec = min(candidates)   # smallest decay that clears the bar
    print(f"\nverdict: RECOMMEND p_decay={rec} -- new-regime adaptation "
          f"{base_rev:.2f} -> {adapt[rec][-1]:.2f}; cost: stationary prediction "
          f"{stationary[0.0][0]:.2f} -> {stationary[rec][0]:.2f}, "
          f"generation {stationary[0.0][1]:.2f} -> {stationary[rec][1]:.2f}")
else:
    print("\nverdict: no decay rate cleared the bar -- inertia is not (only) in P; "
          "investigate slot/embedding pathway")
