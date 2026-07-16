"""
PHASE 35 -- HIERARCHICAL RECALL: does routing generation through a cheap
CATEGORY-level choice first, then resolving to the specific word-level
attractor only within that category, rescue the Attractor Crowding
Collapse phase 34 measured?

Mechanism under test: phase 34 showed grammaticality collapsing as
vocabulary grew at fixed field size (N=128) -- 0.994 -> 0.284 from 50 to
800 words. The working theory (see ROADMAP) is that FLAT recall picks one
winner out of ALL K stored word-attractors at every step, and that K-way
competition gets more failure-prone as K grows, then compounds because
each hop's error becomes the next hop's starting point. This phase tests
a two-stage fix at INFERENCE time only (no retraining, same learned
Hebbian weights): first choose among ~5 CATEGORY attractors (cheap, K
stays 5 no matter the vocabulary), then choose only among the ~K/5 word
attractors that belong to the winning category. If crowding-during-
selection is really what's failing, cutting per-step competition from K
candidates to K/5 candidates should measurably help even before the field
size itself changes.

Caveat, pre-registered: category membership here is read from the corpus
generator's ground truth (an oracle), not discovered unsupervised.
discover_categories_v2 (polysemy_organism.py) exists for that and is the
honest next step if this mechanism proves out -- this phase isolates
whether hierarchical routing helps AT ALL before paying for the
unsupervised-discovery machinery on top of it.
"""

import time
import numpy as np
from organism import Organism, normalize
from phase34_capacity_scaling import (
    build_vocab, sample_stream, grammaticality,
    N_CATS, DIM, NORM, HOLD, STEPS_PER_WORD,
)

rng = np.random.default_rng(7)


def train_and_route(n_words, per_cat, cat, next_cat, emb, train_seq, seed):
    K = max(8, int(1.2 * n_words))
    org = Organism(N=DIM, K=K, omega=0.15, beta=10.0, seed=seed, backend="auto")
    full_stream = []
    for w in train_seq:
        s = normalize(emb[w].astype(complex), NORM)
        full_stream.extend([s] * HOLD)
    org.perceive(full_stream, g_in=5.0, dt=0.05, eta=0.02, recruit=0.45)
    org.consolidate(merge_thresh=0.86, prune_frac=0.015)
    M = org.mem
    n_mem = M.shape[0]

    states = np.array([normalize(emb[w].astype(complex), NORM) for w in train_seq])
    assigns = np.abs((M.conj() @ states.T) / DIM).argmax(0)
    slot_to_word = {}
    for k in range(n_mem):
        members = np.array(train_seq)[assigns == k]
        if len(members):
            slot_to_word[k] = int(np.bincount(members, minlength=n_words).argmax())
    coverage = len(set(slot_to_word.values()))
    slot_cat = {k: cat[w] for k, w in slot_to_word.items()}
    return org, n_mem, slot_to_word, slot_cat, coverage


def flat_recall_grammaticality(org, slot_to_word, cat, next_cat):
    org.beta = 20
    slot_seq = org.recall(steps=20000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
    gen = [slot_to_word[int(s)] for s in slot_seq if int(s) in slot_to_word][:1000]
    return grammaticality(gen, cat, next_cat) if len(gen) > 10 else 0.0


def hierarchical_recall_grammaticality(org, n_mem, slot_to_word, slot_cat, cat, next_cat, seed):
    """Two-stage sampler, inference-time only, same learned org.Pn counts.
    Stage A: category transition matrix (aggregate word-slot Hebbian counts
    by category) -- a flat K_cat=5 competition regardless of vocabulary.
    Stage B: within the winning category, weight candidate word-slots by
    the SOURCE slot's own learned counts toward them (falls back to
    uniform if the source slot never observed a transition into that
    category, e.g. it was pruned or never visited during training)."""
    local_rng = np.random.default_rng(seed)
    cats_by_slot = np.array([slot_cat.get(k, -1) for k in range(n_mem)])
    P_cat = np.zeros((N_CATS, N_CATS))
    for k in range(n_mem):
        if slot_cat.get(k) is None:
            continue
        row = org.Pn[k]
        for j in range(n_mem):
            if slot_cat.get(j) is not None and row[j] > 0:
                P_cat[slot_cat[k], slot_cat[j]] += row[j]
    row_sums = P_cat.sum(1, keepdims=True)
    P_cat = np.divide(P_cat, row_sums, out=np.full_like(P_cat, 1.0 / N_CATS), where=row_sums > 0)

    cat_members = {c: [k for k in range(n_mem) if slot_cat.get(k) == c] for c in range(N_CATS)}

    cur = next(iter(slot_to_word))
    gen = []
    for _ in range(1000):
        cur_cat = slot_cat.get(cur, int(local_rng.integers(N_CATS)))
        next_cat_choice = int(local_rng.choice(N_CATS, p=P_cat[cur_cat]))
        candidates = cat_members[next_cat_choice]
        if not candidates:
            cur = int(local_rng.integers(n_mem))
            continue
        weights = np.array([max(org.Pn[cur, c], 1e-6) for c in candidates])
        weights = weights / weights.sum()
        nxt = candidates[int(local_rng.choice(len(candidates), p=weights))]
        if nxt in slot_to_word:
            gen.append(slot_to_word[nxt])
        cur = nxt
    return grammaticality(gen, cat, next_cat) if len(gen) > 10 else 0.0


def main():
    sweep = [50, 100, 200, 400, 800]
    print("=" * 86)
    print("PHASE 35 -- HIERARCHICAL RECALL vs FLAT RECALL (same trained weights, field N=128 fixed)")
    print("=" * 86)
    header = f"{'vocab':>6} {'flat_gram':>10} {'hier_gram':>10} {'oracle':>7} {'delta':>8} {'sec':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for n_words in sweep:
        t0 = time.time()
        n_words, per_cat, cat, next_cat, emb = build_vocab(n_words)
        seq_len = STEPS_PER_WORD * n_words
        train_seq = sample_stream(n_words, per_cat, cat, next_cat, seq_len, seed=99)

        org, n_mem, slot_to_word, slot_cat, coverage = train_and_route(
            n_words, per_cat, cat, next_cat, emb, train_seq, seed=0)

        flat_gram = flat_recall_grammaticality(org, slot_to_word, cat, next_cat)
        hier_gram = hierarchical_recall_grammaticality(
            org, n_mem, slot_to_word, slot_cat, cat, next_cat, seed=1)

        oracle_seq = sample_stream(n_words, per_cat, cat, next_cat, 1000, seed=200)
        gram_oracle = grammaticality(oracle_seq, cat, next_cat)

        sec = time.time() - t0
        delta = hier_gram - flat_gram
        rows.append((n_words, flat_gram, hier_gram, gram_oracle, delta))
        print(f"{n_words:>6} {flat_gram:>10.3f} {hier_gram:>10.3f} {gram_oracle:>7.3f} "
              f"{delta:>+8.3f} {sec:>8.1f}")

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    first, last = rows[0], rows[-1]
    print(f"Flat recall:          {first[1]:.3f} @ {first[0]} words -> {last[1]:.3f} @ {last[0]} words")
    print(f"Hierarchical recall:  {first[2]:.3f} @ {first[0]} words -> {last[2]:.3f} @ {last[0]} words")
    print(f"Gap closed at {last[0]} words: hierarchical recovers "
          f"{(last[2]-last[1])/max(last[3]-last[1],1e-9)*100:.0f}% of the flat-vs-oracle gap "
          f"({last[1]:.3f} -> {last[2]:.3f}, oracle {last[3]:.3f}).")
    improves = all(r[4] >= -0.02 for r in rows)
    grows = last[4] > first[4] + 0.05
    if last[4] > 0.15:
        print("\n-> Hierarchical routing materially rescues quality at scale: crowding during "
              "SELECTION, not the stored attractors themselves, is a load-bearing cause of the "
              "collapse. Worth productizing (category discovery via discover_categories_v2 "
              "instead of the oracle labels used here) as the first real fix.")
    elif last[4] > 0.03:
        print("\n-> Hierarchical routing helps some but does not close the gap: selection "
              "crowding is A cause, not the whole story -- the embedding/attractor crowding "
              "measured in phase 34 (NN similarity rising with vocab) likely still contributes.")
    else:
        print("\n-> Hierarchical routing does not meaningfully help: the collapse is not "
              "primarily a selection-time artifact. Suspect the attractors themselves "
              "(and/or the sparse per-pair Hebbian statistics) rather than the K-way choice.")


if __name__ == "__main__":
    main()
