"""
PHASE 13 -- RECALL DYNAMICS OVERHAUL: one subsystem, three symptoms

Phases 11-12 and the deployment benchmark left three open gaps that all
point at recall's selection dynamics, not at memory:

  1. CROWDING COLLAPSE: at V=300 with structured embeddings, storage is
     perfect (coverage/purity 1.00, prediction 0.82) but generation scores
     0.05 -- far below the 0.33 chance floor.
  2. NOISE FRAGILITY: sigma=0.3 input noise leaves 92% coverage but drops
     generation to 0.19.
  3. PLATEAU: even in the clean small world, generation saturates at ~0.62
     against the 0.88 ceiling, no matter how much data.

Diagnosis (measured below): the failures are (a) UNBOUNDED COMPETITION --
recall's softmax lets every memory pull on the field, so crowded
same-category attractors blend into a blob and the state flickers among
them (the sub-chance 0.05 is within-category hopping, which the cyclic
grammar scores as always-wrong); and (b) FLICKER COUNTING -- a transition
is recorded the moment any memory's overlap crosses 0.5, so mid-flight
states between attractors are logged as hops.

Fixes in organism.recall2(), each organism-native and each ablatable:
  - topk lateral inhibition: only the k best-scoring memories compete for
    the field pull. Biology: local inhibitory circuits enforce sparse
    winner competition.
  - hop commitment: a transition counts only when the new memory exceeds
    `commit` overlap AND stays argmax for `debounce` consecutive steps.
    Biology: attractor transitions are events, not sample-by-sample noise.

Protocol: each failing case, four variants on the SAME trained organism --
  baseline   : recall()                      (the phase-2 dynamics)
  commit-only: recall2(topk=all, debounce=20)
  topk-only  : recall2(topk=8, debounce=1, commit=0.5)
  both       : recall2(topk=8, debounce=20)
Scores: grammaticality, within-category hop fraction (the crowding
signature), junk-slot hop fraction (the noise signature), hops emitted.
"""

import numpy as np
from organism import Organism, normalize

ANIMAL, ACTION, OBJECT = 0, 1, 2
NEXT_FWD = {0: 1, 1: 2, 2: 0}
P_CORRECT = 0.88
HOLD = 12


def build_small_world():
    ANIMALS = ['cat','dog','bird','horse','cow','pig','sheep','wolf']
    ACTIONS = ['run','jump','swim','eat','sleep','hunt','hide','play']
    OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
    vocab = ANIMALS + ACTIONS + OBJECTS
    V = len(vocab)
    cats = np.array([ANIMAL]*8 + [ACTION]*8 + [OBJECT]*10)
    local = np.random.default_rng(13)
    N = 30
    cat_bases = np.zeros((3, N))
    cat_bases[0, 0:3] = 1.0; cat_bases[1, 3:6] = 1.0; cat_bases[2, 6:9] = 1.0
    emb = np.zeros((V, N))
    for i in range(V):
        emb[i] = 0.6*cat_bases[cats[i]] + 0.4*local.standard_normal(N)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb, cats, N


def build_big_world(V=300, N=128, base_w=0.6):
    local = np.random.default_rng(13)
    cats = np.array([i % 3 for i in range(V)])
    third = max(N // 10, 3)
    cat_bases = np.zeros((3, N))
    cat_bases[0, 0:third] = 1.0
    cat_bases[1, third:2*third] = 1.0
    cat_bases[2, 2*third:3*third] = 1.0
    emb = np.zeros((V, N))
    for i in range(V):
        emb[i] = base_w*cat_bases[cats[i]] + (1-base_w)*local.standard_normal(N)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb, cats, N


def sample_stream(n, cats, seed):
    local = np.random.default_rng(seed)
    pools = [np.where(cats == c)[0] for c in range(3)]
    cat = ANIMAL
    seq = []
    for _ in range(n):
        seq.append(int(local.choice(pools[cat])))
        cat = NEXT_FWD[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != NEXT_FWD[cat]]))
    return seq


def train(emb, cats, N, n_words, noise=0.0, seed=99):
    NORM = np.sqrt(N)
    V = len(emb)
    local = np.random.default_rng(1)
    seq = sample_stream(n_words, cats, seed)
    stream = []
    for w in seq:
        e = emb[w] + (noise * local.standard_normal(N) if noise > 0 else 0)
        s = normalize(e.astype(complex), NORM)
        stream.extend([s] * HOLD)
    org = Organism(N=N, K=int(V*1.25)+10, omega=0.15, beta=10.0, seed=0)
    org.perceive(stream, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    org.consolidate(merge_thresh=0.84, prune_frac=0.02)
    return org


def slot_maps(org, emb, cats, N):
    NORM = np.sqrt(N)
    V = len(emb)
    states = np.array([normalize(emb[w].astype(complex), NORM) for w in range(V)])
    M = org.mem
    ov = np.abs((M.conj() @ states.T) / N)
    s2w = {i: int(np.argmax(ov[i])) for i in range(M.shape[0])}
    dedicated = set(int(ov[:, w].argmax()) for w in range(V))   # some word's best slot
    return s2w, dedicated


def score(slot_seq, s2w, dedicated, cats, n_gen=400):
    seq = [int(s) for s in slot_seq][:n_gen]
    if len(seq) < 2:
        return dict(gram=0.0, within=0.0, junk=0.0, hops=len(seq))
    junk = np.mean([s not in dedicated for s in seq])
    gen = [s2w[s] for s in seq]
    pairs = list(zip(gen[:-1], gen[1:]))
    gram = np.mean([NEXT_FWD[cats[a]] == cats[b] for a, b in pairs])
    within = np.mean([cats[a] == cats[b] for a, b in pairs])
    return dict(gram=gram, within=within, junk=junk, hops=len(seq))


def run_case(name, org, emb, cats, N, steps=60000):
    s2w, dedicated = slot_maps(org, emb, cats, N)
    Ku = org.mem.shape[0]
    variants = [
        ("baseline (recall)",  lambda: org.recall(steps=steps)),
        ("commit-only",        lambda: org.recall2(steps=steps, topk=Ku, debounce=20)),
        ("topk-only",          lambda: org.recall2(steps=steps, topk=8, debounce=1, commit=0.5)),
        ("both",               lambda: org.recall2(steps=steps, topk=8, debounce=20)),
    ]
    print(f"\n=== {name}  ({Ku} memories) ===")
    print(f"{'variant':<20} {'grammaticality':>15} {'within-cat hops':>16} "
          f"{'junk hops':>10} {'hops':>6}")
    out = {}
    for label, fn in variants:
        r = score(fn(), s2w, dedicated, cats)
        out[label] = r
        print(f"{label:<20} {r['gram']:>15.3f} {r['within']:>16.3f} "
              f"{r['junk']:>10.3f} {r['hops']:>6}")
    return out


print("Training the three failing cases (perceive once, recall variants share it)")
emb_s, cats_s, N_s = build_small_world()
res = {}

org = train(emb_s, cats_s, N_s, 4000)
res['plateau'] = run_case("1. PLATEAU -- clean small world (26 words)", org, emb_s, cats_s, N_s)

org = train(emb_s, cats_s, N_s, 4000, noise=0.3)
res['noise'] = run_case("2. NOISE -- small world, sigma=0.3 input corruption", org, emb_s, cats_s, N_s)

emb_b, cats_b, N_b = build_big_world()
org = train(emb_b, cats_b, N_b, 3000)
res['crowding'] = run_case("3. CROWDING -- V=300 structured embeddings", org, emb_b, cats_b, N_b)

print("\n" + "="*70)
print("PHASE 13 SUMMARY -- grammaticality (oracle 0.88, chance ~0.33)")
print(f"{'case':<12} {'baseline':>9} {'commit':>8} {'topk':>7} {'both':>7}")
for case in ('plateau', 'noise', 'crowding'):
    r = res[case]
    print(f"{case:<12} {r['baseline (recall)']['gram']:>9.3f} "
          f"{r['commit-only']['gram']:>8.3f} {r['topk-only']['gram']:>7.3f} "
          f"{r['both']['gram']:>7.3f}")

wins = sum(1 for case in res
           if res[case]['both']['gram'] > res[case]['baseline (recall)']['gram'] + 0.05)
best_all = all(res[case]['both']['gram'] >= 0.6 for case in res)
if wins == 3 and best_all:
    print("\nverdict: recall2 (lateral inhibition + hop commitment) fixes all three"
          "\n         symptoms -- adopt as the default recall going forward")
elif wins >= 2:
    print(f"\nverdict: recall2 improves {wins}/3 cases -- adopt where it wins; "
          "the remaining case needs a different mechanism (see per-case tables)")
else:
    print("\nverdict: the recall hypothesis is wrong or the fixes are miscalibrated"
          " -- preserve as negative result and re-diagnose")
