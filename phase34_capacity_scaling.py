"""
PHASE 34 -- CAPACITY/QUALITY SCALING: holding the field's compute budget
FIXED (N=128, the same "hardware"), how does generation quality degrade as
vocabulary size grows, and how does that compare to a compute-matched
conventional baseline (a bigram frequency table)?

Motivating question (owner, 2026-07-15): can the oscillator field reach
LLM-level quality without LLM-level compute? Phase 20 (547K-word real text)
and Phase 33 (K=40 slot flooding) already hinted the field's capacity, not
its compute-per-step, is the binding constraint. This phase makes that a
controlled, swept measurement instead of an inference from two unrelated
runs.

Protocol: same cyclic 5-category grammar as Phase 5 (p_correct=0.85), swept
across vocabulary size (words per category held fixed at 1/5 of total).
Per-word training exposure is held CONSTANT across the sweep (steps scale
with vocab) so any quality drop is attributable to capacity, not under-
training. Field size N=128 and slot budget K=1.2x vocab are held fixed in
spirit -- K must grow with vocab (you cannot store more concepts than you
have slots for), but N (the per-step compute cost) does not, which is the
actual "same hardware" condition worth testing.

Baseline: a bigram frequency table over word IDs, trained on the identical
stream. Bigram tables are the cheapest possible sequence model -- if they
match or beat the organism's grammaticality at a fraction of the wall-clock
cost, that is the honest answer to "can we skip the compute."
"""

import time
import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(42)

N_CATS = 5
DIM = 128
NORM = np.sqrt(DIM)
P_CORRECT = 0.85
STEPS_PER_WORD = 200   # fixed training exposure per vocabulary item
HOLD = 10              # ticks per stream token (as phase 5)

CAT_NAMES = ['SUBJ', 'VERB', 'OBJ', 'PLACE', 'TIME']


def build_vocab(n_words):
    per_cat = n_words // N_CATS
    n_words = per_cat * N_CATS
    words = list(range(n_words))
    cat = {w: w // per_cat for w in words}
    next_cat = {c: (c + 1) % N_CATS for c in range(N_CATS)}

    cat_bases = np.zeros((N_CATS, DIM))
    for c in range(N_CATS):
        cat_bases[c, c * 3:(c + 1) * 3] = 1.0
    emb = np.zeros((n_words, DIM))
    for w in words:
        emb[w] = 0.55 * cat_bases[cat[w]] + 0.45 * rng.standard_normal(DIM)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return n_words, per_cat, cat, next_cat, emb


def sample_stream(n_words, per_cat, cat, next_cat, n_steps, seed):
    local_rng = np.random.default_rng(seed)
    c = 0
    w = int(local_rng.integers(per_cat))
    seq = [w]
    for _ in range(n_steps - 1):
        if local_rng.random() < P_CORRECT:
            c = next_cat[c]
        else:
            c = int(local_rng.choice([x for x in range(N_CATS) if x != next_cat[cat[w]]]))
        w = c * per_cat + int(local_rng.integers(per_cat))
        seq.append(w)
    return seq


def grammaticality(wseq, cat, next_cat):
    ok = tot = 0
    for a, b in zip(wseq[:-1], wseq[1:]):
        if next_cat[cat[a]] == cat[b]:
            ok += 1
        tot += 1
    return ok / tot if tot else 0.0


def run_bigram_baseline(train_seq, n_words, cat, next_cat, seed):
    t0 = time.time()
    counts = np.zeros((n_words, n_words))
    for a, b in zip(train_seq[:-1], train_seq[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(1, keepdims=True)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    local_rng = np.random.default_rng(seed)
    w = train_seq[-1]
    gen = []
    for _ in range(1000):
        row = P[w]
        if row.sum() > 0:
            w = int(local_rng.choice(n_words, p=row))
        else:
            w = int(local_rng.integers(n_words))
        gen.append(w)
    train_time = time.time() - t0
    return grammaticality(gen, cat, next_cat), train_time, counts.nbytes


def run_organism(train_seq, n_words, cat, next_cat, emb, seed):
    K = max(8, int(1.2 * n_words))
    t0 = time.time()
    org = Organism(N=DIM, K=K, omega=0.15, beta=10.0, seed=seed, backend="auto")
    stream = []
    for w in train_seq:
        s = normalize(emb[w].astype(complex), NORM)
        stream.extend([s] * HOLD)
    org.perceive(stream, g_in=5.0, dt=0.05, eta=0.02, recruit=0.45)
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

    org.beta = 20
    slot_seq = org.recall(steps=20000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
    gen = [slot_to_word[int(s)] for s in slot_seq if int(s) in slot_to_word][:1000]
    train_time = time.time() - t0
    gram = grammaticality(gen, cat, next_cat) if len(gen) > 10 else 0.0
    state_bytes = M.nbytes + org.Pn.nbytes
    return gram, train_time, state_bytes, n_mem, coverage


def main():
    sweep = [50, 100, 200, 400, 800]
    print("=" * 78)
    print("PHASE 34 -- CAPACITY/QUALITY SCALING (field N=128 fixed across sweep)")
    print("=" * 78)
    header = (f"{'vocab':>6} {'org_gram':>9} {'big_gram':>9} {'oracle':>7} "
              f"{'org_cov':>8} {'org_mem':>8} {'org_sec':>8} {'big_sec':>8} "
              f"{'org_KB':>8} {'big_KB':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for n_words in sweep:
        n_words, per_cat, cat, next_cat, emb = build_vocab(n_words)
        seq_len = STEPS_PER_WORD * n_words
        train_seq = sample_stream(n_words, per_cat, cat, next_cat, seq_len, seed=99)

        org_gram, org_sec, org_bytes, n_mem, coverage = run_organism(
            train_seq, n_words, cat, next_cat, emb, seed=0)
        big_gram, big_sec, big_bytes = run_bigram_baseline(train_seq, n_words, cat, next_cat, seed=1)

        oracle_seq = sample_stream(n_words, per_cat, cat, next_cat, 1000, seed=200)
        gram_oracle = grammaticality(oracle_seq, cat, next_cat)

        rows.append((n_words, org_gram, big_gram, gram_oracle, coverage, n_mem,
                    org_sec, big_sec, org_bytes, big_bytes))
        print(f"{n_words:>6} {org_gram:>9.3f} {big_gram:>9.3f} {gram_oracle:>7.3f} "
              f"{coverage:>4}/{n_words:<3} {n_mem:>8} {org_sec:>8.2f} {big_sec:>8.4f} "
              f"{org_bytes/1024:>8.1f} {big_bytes/1024:>8.1f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    first, last = rows[0], rows[-1]
    print(f"Organism grammaticality: {first[1]:.3f} @ {first[0]} words -> "
          f"{last[1]:.3f} @ {last[0]} words  (oracle ~{last[3]:.3f})")
    print(f"Bigram grammaticality:   {first[2]:.3f} @ {first[0]} words -> "
          f"{last[2]:.3f} @ {last[0]} words  (oracle ~{last[3]:.3f})")
    print(f"Organism compute cost grew {last[6]/first[6]:.1f}x (train+recall wall time) "
          f"over a {last[0]/first[0]:.0f}x vocab increase, at FIXED field size N={DIM}.")
    print(f"Bigram compute cost grew {last[7]/first[7]:.1f}x over the same vocab increase.")
    coverage_frac_first = first[4] / first[0]
    coverage_frac_last = last[4] / last[0]
    print(f"Slot coverage (memories formed / vocab): {coverage_frac_first:.2f} @ {first[0]} words "
          f"-> {coverage_frac_last:.2f} @ {last[0]} words.")
    if last[1] < first[1] - 0.05 or coverage_frac_last < coverage_frac_first - 0.05:
        print("\n-> Quality/coverage degrades as vocabulary grows at fixed N: capacity, "
              "not per-step compute, is the binding constraint (confirms phase 20/33).")
    else:
        print("\n-> No material degradation observed in this range -- capacity ceiling, "
              "if any, lies beyond the tested vocabulary sizes.")
    if last[2] >= last[1] - 0.02:
        print(f"-> The bigram baseline matches or beats the organism's grammaticality "
              f"at {last[6]/max(last[7],1e-9):.0f}x less wall-clock cost: for THIS task "
              f"(sequence grammar), conventional cheap methods are not being beaten, "
              f"let alone LLM-level ones.")


if __name__ == "__main__":
    main()
