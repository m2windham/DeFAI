"""
PHASE 12 -- CAPACITY SCALING: does the slot architecture survive 1000 concepts?

The single most informative experiment for the Tier-3 question ("could this
architecture ever host language-scale vocabularies?"). Everything so far ran
at 26-30 words; phase 5 stopped at 100. The scaling law is unknown -- this
measures it.

Setup: the standard cyclic 3-category grammar, vocabulary V swept over
{30, 100, 300, 1000} at field size N=128, plus a field-size sweep
{64, 128, 256} at V=300. Stream length 10 occurrences per word. K = V + 25%
headroom. p_decay from phase 11 is NOT used (stationary world).

Two embedding regimes, because "capacity" confounds two different limits:
  - STRUCTURED (0.6 category base + 0.4 noise, as all prior phases): words
    within a category crowd each other; measures capacity under realistic
    correlated inputs.
  - SPREAD (0.3 base + 0.7 noise): near-orthogonal words; measures the
    architecture's intrinsic slot capacity with minimal crowding.

Metrics per config:
  coverage   : fraction of words owning a dedicated slot (their best slot
               maps back to them) -- the vocabulary actually stored.
  purity     : fraction of used slots that are some word's best slot --
               1 - purity is wasted/duplicate slots.
  crosstalk  : mean over words of the SECOND-best slot overlap -- how close
               the nearest interfering memory sits.
  prediction : next-category accuracy on a held-out probe (structure
               learning, cheap at any scale).
  time/1k    : wall-clock seconds per 1000 words of stream (perceive only).

Generation (recall) is evaluated only up to V=300: recall wall-clock scales
with K and the point of this phase is storage/structure capacity, not
generation speed.

RESULT (recorded from the committed run):
  - STORAGE SCALES. Coverage and slot purity are 1.00 at every vocabulary
    size up to V=1000, in BOTH embedding regimes (997-1001 slots recruited
    for 1000 words). The slot mechanism is not the capacity bottleneck at
    this scale. Crosstalk grows slowly (0.30 -> 0.39 structured).
  - PREDICTION DEGRADES GRACEFULLY: 0.885 -> 0.786 from V=30 to V=1000
    (ceiling 0.88).
  - GENERATION is the scaling casualty, and the two regimes localize it:
    structured embeddings collapse recall at V=300 (0.05) while SPREAD
    embeddings generate excellently at the same size (0.927). The limit is
    representational crowding (within-category attractor proximity
    interacting with the 0.84 consolidation merge and recall dynamics tuned
    for small K), NOT slot count.
  - FIELD SIZE scales sublinearly: N=128 suffices for V=1000; N=64 loses
    coverage (0.95) at V=300.
  - Wall-clock: 1000-word vocabulary learns a 10k-word stream in about a
    minute on CPU. (Per-config timings in the log are contaminated by a
    concurrent run for the first few rows -- treat as indicative only.)
  - Tier-3 implication: vocabulary storage is a non-problem; the work is in
    generation-under-crowding and the binding/hierarchy list.
"""

import time
import numpy as np
from organism import Organism, normalize

ANIMAL, ACTION, OBJECT = 0, 1, 2
NEXT_FWD = {0: 1, 1: 2, 2: 0}
P_CORRECT = 0.88
HOLD = 12


def build_world(V, N, base_w, seed=13):
    """V words split evenly over 3 categories; returns embeddings and cats."""
    local = np.random.default_rng(seed)
    cats = np.array([i % 3 for i in range(V)])
    cat_bases = np.zeros((3, N))
    third = max(N // 10, 3)
    cat_bases[0, 0:third] = 1.0
    cat_bases[1, third:2*third] = 1.0
    cat_bases[2, 2*third:3*third] = 1.0
    emb = np.zeros((V, N))
    for i in range(V):
        emb[i] = base_w*cat_bases[cats[i]] + (1-base_w)*local.standard_normal(N)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb, cats


def sample_stream(n, V, cats, seed):
    local = np.random.default_rng(seed)
    pools = [np.where(cats == c)[0] for c in range(3)]
    cat = ANIMAL
    seq = []
    for _ in range(n):
        seq.append(int(local.choice(pools[cat])))
        cat = NEXT_FWD[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != NEXT_FWD[cat]]))
    return seq


def run_config(V, N, base_w, label, do_recall):
    NORM = np.sqrt(N)
    emb, cats = build_world(V, N, base_w)
    K = int(V * 1.25)
    n_words = 10 * V
    seq = sample_stream(n_words, V, cats, seed=99)
    org = Organism(N=N, K=K, omega=0.15, beta=10.0, seed=0)

    states = np.array([normalize(emb[w].astype(complex), NORM) for w in range(V)])
    stream = []
    for w in seq:
        stream.extend([states[w]] * HOLD)

    t0 = time.time()
    org.perceive(stream, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    dt_perceive = time.time() - t0
    per_1k = dt_perceive / (n_words / 1000)

    M = org.xi[org.used]
    used_idx = np.where(org.used)[0]
    ov = np.abs((M.conj() @ states.T) / N)              # slots x words
    best_slot = ov.argmax(0)                            # per word
    slot_best_word = ov.argmax(1)                       # per slot
    coverage = np.mean([slot_best_word[best_slot[w]] == w for w in range(V)])
    purity = len(set(int(best_slot[w]) for w in range(V)
                     if slot_best_word[best_slot[w]] == w)) / max(len(used_idx), 1)
    # crosstalk: second-best slot overlap per word
    if ov.shape[0] > 1:
        part = np.partition(ov, -2, axis=0)
        crosstalk = float(part[-2].mean())
    else:
        crosstalk = 0.0

    # next-category prediction on a held-out probe
    probe = sample_stream(1000, V, cats, seed=7)
    w2s = {w: int(used_idx[best_slot[w]]) for w in range(V)}
    s2w = {int(used_idx[i]): int(slot_best_word[i]) for i in range(len(used_idx))}
    ok = tot = 0
    for a, b in zip(probe[:-1], probe[1:]):
        row = org.P[w2s[a]]
        if row.sum() <= 0: continue
        j = int(np.argmax(row))
        if j in s2w:
            ok += cats[s2w[j]] == cats[b]; tot += 1
    pred = ok / max(tot, 1)

    gram = float('nan')
    if do_recall:
        org.consolidate(merge_thresh=0.84, prune_frac=0.02)
        Mc = org.mem
        ovc = np.abs((Mc.conj() @ states.T) / N)
        s2wc = {i: int(np.argmax(ovc[i])) for i in range(Mc.shape[0])}
        gen = [s2wc[int(s)] for s in org.recall(steps=40000)][:400]
        okg = sum(1 for a, b in zip(gen[:-1], gen[1:])
                  if NEXT_FWD[cats[a]] == cats[b])
        gram = okg / max(len(gen)-1, 1)

    print(f"  {label:<28} coverage={coverage:.2f}  purity={purity:.2f}  "
          f"crosstalk={crosstalk:.2f}  prediction={pred:.3f}  "
          f"gen={'--' if np.isnan(gram) else f'{gram:.3f}'}  "
          f"time/1k words={per_1k:.1f}s  slots used={len(used_idx)}/{K}")
    return coverage, purity, crosstalk, pred, gram, per_1k


print("="*70)
print("VOCABULARY SWEEP at N=128 (stream = 10 occurrences/word)")
print("\n-- STRUCTURED embeddings (0.6 category base: realistic crowding) --")
res_struct = {}
for V in (30, 100, 300, 1000):
    res_struct[V] = run_config(V, 128, 0.6, f"V={V}", do_recall=(V <= 300))

print("\n-- SPREAD embeddings (0.3 base: near-orthogonal, intrinsic capacity) --")
res_spread = {}
for V in (30, 100, 300, 1000):
    res_spread[V] = run_config(V, 128, 0.3, f"V={V}", do_recall=(V <= 300))

print("\n" + "="*70)
print("FIELD-SIZE SWEEP at V=300 (structured embeddings)")
for N in (64, 128, 256):
    run_config(300, N, 0.6, f"N={N}", do_recall=False)

print("\n" + "="*70)
print("PHASE 12 SUMMARY -- coverage vs vocabulary (N=128)")
print(f"{'V':>6} {'structured':>11} {'spread':>8}")
for V in (30, 100, 300, 1000):
    print(f"{V:>6} {res_struct[V][0]:>11.2f} {res_spread[V][0]:>8.2f}")
print("\nInterpretation guide: if SPREAD stays high while STRUCTURED falls,")
print("the limit is representational crowding (embedding geometry), not the")
print("slot mechanism; if both fall together, the architecture itself is the")
print("bottleneck at that scale.")
