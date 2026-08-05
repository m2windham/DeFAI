"""
PHASE 36 -- UNSUPERVISED HIERARCHICAL RECALL: does phase 35's rescue of the
Attractor Crowding Collapse survive when the category labels are DISCOVERED
by the organism instead of read from the corpus generator's ground truth?

Phase 35 showed hierarchical (category-then-word) recall holds ~0.84-0.85
grammaticality flat from 50 to 800 words while flat recall collapses
0.994 -> 0.284 -- but its router was handed oracle category membership, a
pre-registered caveat. This phase closes that caveat: category membership
now comes from discover_categories_v2 (polysemy_organism.py) -- PPMI-
transformed transition profiles of the organism's OWN learned slot-level
counts, k-means clustered -- with k selected by phase 24's DISTINCTNESS
criterion (max over k of the minimum pairwise distance between category
transition profiles; the parsimony selector that replaced silhouette-argmax
after phase 24 measured silhouette flat and predictive criteria
monotonically over-splitting). Validity is certified per row by phase 24's
MI-vs-permutation-null: class-bigram mutual information of the discovered
partition against 500 label permutations at fixed category sizes.

Everything upstream is IDENTICAL to phases 34/35: same corpus generator
(cyclic 5-category grammar, p_correct=0.85), same sweep (50 -> 800 words at
fixed field N=128), same training (exposure per word held constant), same
scorers (grammaticality against the generator's transition rule -- ground
truth for EVALUATION only, per standing rules), same flat-recall arm, and
the oracle-labeled hierarchical arm is re-run in-process as both a phase-35
reproduction check and the ceiling the discovered labels are measured
against. Zero labels enter the mechanism anywhere: discovery sees only the
organism's own transition counts; the distinctness selector and MI
certificate are computed from those same counts.

Pre-registered predictions (ROADMAP row 36, fixed before this run):
  (a) if discovered categories are near-oracle quality, hierarchical recall
      retains most of the flat-vs-oracle gap recovery, degraded in
      proportion to category impurity;
  (b) the honest negative is a PURITY FLOOR below which impure categories
      mis-route stage-B resolution and hierarchical recall loses to flat --
      measured either way.
Concrete bands, pre-registered for this protocol: with recovery ratio
rho = (hier_disc - flat) / (hier_oracle - flat) at 800 words,
  rho >= 0.7  -> un-oracled fix CONFIRMED (caveat closed);
  0.3-0.7     -> PARTIAL: real but impurity-degraded -- check that the
                 per-row oracle-minus-discovered deficit tracks (1-purity);
  rho < 0.3, or hier_disc < flat at >= 200 words -> NEGATIVE: discovery
                 quality is the binding constraint; report the purity floor.
The known small-vocab hierarchy tax (phase 35: 0.994 -> 0.839 at 50 words,
crossover between 100 and 200) is expected to persist and does not count
against the mechanism.

One asymmetry, noted: the oracle router can only label slots that map to a
word (slot -> word -> true category), while discovery labels EVERY kept
slot from its transition profile -- unattributed slots participate in
discovered routing. That is not a bug in the comparison; it is the honest
deployment condition (a product has no word attribution to lean on).
"""

import time
import numpy as np
from organism import normalize
from polysemy_organism import PolysemyOrganism
from phase34_capacity_scaling import (
    build_vocab, sample_stream, grammaticality,
    N_CATS, DIM, NORM, HOLD, STEPS_PER_WORD,
)
from phase35_hierarchical_recall import (
    flat_recall_grammaticality,
    hierarchical_recall_grammaticality as phase35_hier,
)

K_RANGE = range(2, 13)      # candidate category counts for discovery
KMEANS_SEED = 3             # same clustering seed convention as phases 23/24
N_NULL = 500                # permutation draws for the MI certificate

# phase 35's committed anchors (ROADMAP row 35) -- reproduction check only
P35_ANCHORS = {50: dict(flat=0.994, hier=0.839), 800: dict(flat=0.284, hier=0.845, oracle=0.854)}


def train_and_route(n_words, cat, emb, train_seq, seed):
    """Identical training/attribution to phase 35's train_and_route, with a
    PolysemyOrganism (adds discovery state only -- no extra rng draws, same
    perceive/consolidate path) and two bounded-memory changes that leave
    every value identical: the training stream is fed as a generator (E2's
    chunked streaming; the materialized list is ~3.4 GB at 800 words) and
    the slot-attribution matmul runs in column chunks."""
    K = max(8, int(1.2 * n_words))
    org = PolysemyOrganism(N=DIM, K=K, omega=0.15, beta=10.0, seed=seed, backend="auto")

    def frames():
        for w in train_seq:
            s = normalize(emb[w].astype(complex), NORM)
            for _ in range(HOLD):
                yield s
    org.perceive(frames(), g_in=5.0, dt=0.05, eta=0.02, recruit=0.45)
    org.consolidate(merge_thresh=0.86, prune_frac=0.015)
    M = org.mem
    n_mem = M.shape[0]

    seq_arr = np.array(train_seq)
    assigns = np.empty(len(train_seq), dtype=int)
    CH = 20000
    for i0 in range(0, len(train_seq), CH):
        block = np.array([normalize(emb[w].astype(complex), NORM)
                          for w in train_seq[i0:i0 + CH]])
        assigns[i0:i0 + CH] = np.abs((M.conj() @ block.T) / DIM).argmax(0)
    slot_to_word = {}
    for k in range(n_mem):
        members = seq_arr[assigns == k]
        if len(members):
            slot_to_word[k] = int(np.bincount(members, minlength=n_words).argmax())
    coverage = len(set(slot_to_word.values()))
    slot_cat_oracle = {k: cat[w] for k, w in slot_to_word.items()}
    return org, n_mem, slot_to_word, slot_cat_oracle, coverage


def hier_recall(org, n_mem, slot_to_word, slot_cat, n_cats, cat, next_cat, seed):
    """Phase 35's two-stage router, generalized only in the category count
    (n_cats parameter instead of the module-global N_CATS). At n_cats=5 with
    oracle labels this consumes rng identically to phase 35's function --
    asserted against it at the first sweep point below."""
    local_rng = np.random.default_rng(seed)
    P_cat = np.zeros((n_cats, n_cats))
    for k in range(n_mem):
        if slot_cat.get(k) is None:
            continue
        row = org.Pn[k]
        for j in range(n_mem):
            if slot_cat.get(j) is not None and row[j] > 0:
                P_cat[slot_cat[k], slot_cat[j]] += row[j]
    row_sums = P_cat.sum(1, keepdims=True)
    P_cat = np.divide(P_cat, row_sums, out=np.full_like(P_cat, 1.0 / n_cats), where=row_sums > 0)

    cat_members = {c: [k for k in range(n_mem) if slot_cat.get(k) == c] for c in range(n_cats)}

    cur = next(iter(slot_to_word))
    gen = []
    for _ in range(1000):
        cur_cat = slot_cat.get(cur, int(local_rng.integers(n_cats)))
        next_cat_choice = int(local_rng.choice(n_cats, p=P_cat[cur_cat]))
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


# ---------------------------------------------------------------- discovery
def _slot_cat_bigram(raw_counts, labels, k):
    """M[c,d] = total raw transition count from slots labeled c into slots
    labeled d -- the slot-level analog of phase 24's corpus-pair category
    bigram, built from the organism's own learned counts (label-free)."""
    Z = np.zeros((len(labels), k))
    Z[np.arange(len(labels)), labels] = 1.0
    return Z.T @ raw_counts @ Z


def _distinctness(M):
    """Phase 24's parsimony selector on a category-bigram matrix: minimum
    pairwise distance between category transition profiles (out-row p(.|C)
    concatenated with in-column p(C|.)). Collapses when a split creates a
    near-duplicate category; argmax over k is the largest partition whose
    categories are all still predictively distinct."""
    k = M.shape[0]
    R = M / (M.sum(1, keepdims=True) + 1e-9)
    C = (M / (M.sum(0, keepdims=True) + 1e-9)).T
    P = np.concatenate([R, C], axis=1)
    return float(min(np.linalg.norm(P[i] - P[j])
                     for i in range(k) for j in range(i + 1, k)))


def _mutual_information(M):
    p = M / M.sum()
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = p * np.log2(p / (pi @ pj + 1e-300) + 1e-300)
    return float(term[p > 0].sum())


def discover(org, verbose=True):
    """discover_categories_v2 clustering per candidate k, k selected by
    distinctness-argmax (phase 24), certified by MI-vs-permutation-null at
    the chosen k. Returns compacted labels (empty clusters dropped), the
    effective category count, and the certificate."""
    raw = org.P[np.ix_(org.kept_idx, org.kept_idx)]
    n = org.mem.shape[0]
    cands = []
    for k in K_RANGE:
        if k >= n:
            continue
        res = org.discover_categories_v2(k_range=[k], seed=KMEANS_SEED)
        labels = np.array([res['word_slot_to_cat'][i] for i in range(n)])
        # compact away empty clusters so distinctness compares real categories
        occupied = sorted(set(labels.tolist()))
        remap = {c: i for i, c in enumerate(occupied)}
        labels = np.array([remap[c] for c in labels])
        k_eff = len(occupied)
        if k_eff < 2:
            continue
        d = _distinctness(_slot_cat_bigram(raw, labels, k_eff))
        cands.append((d, k, k_eff, labels, res['silhouette']))
        if verbose:
            print(f"    k={k:2d} (eff {k_eff:2d}): distinctness={d:.4f}  "
                  f"silhouette={res['silhouette']:.3f}")
    d_star, k_star, k_eff, labels, sil = max(cands, key=lambda t: t[0])
    # leave the organism's discovery state on the winning k
    org.discover_categories_v2(k_range=[k_star], seed=KMEANS_SEED)

    M = _slot_cat_bigram(raw, labels, k_eff)
    mi = _mutual_information(M)
    null_rng = np.random.default_rng(11)
    null = np.array([_mutual_information(_slot_cat_bigram(raw, null_rng.permutation(labels), k_eff))
                     for _ in range(N_NULL)])
    z = (mi - null.mean()) / (null.std() + 1e-12)
    p99 = float(np.percentile(null, 99))
    return labels, k_star, k_eff, d_star, mi, p99, z


def purity_and_vmeasure(labels, slot_to_word, slot_cat_oracle):
    """EVALUATION ONLY: quality of the discovered partition against the
    generator's true categories, over slots with a word attribution."""
    attributed = sorted(slot_to_word)
    y_true = np.array([slot_cat_oracle[k] for k in attributed])
    y_disc = np.array([labels[k] for k in attributed])
    pur = 0
    for c in set(y_disc.tolist()):
        members = y_true[y_disc == c]
        pur += np.bincount(members, minlength=N_CATS).max()
    purity = pur / len(attributed)
    try:
        from sklearn.metrics import v_measure_score
        vm = float(v_measure_score(y_true, y_disc))
    except Exception:
        vm = float('nan')
    return float(purity), vm


def main():
    sweep = [50, 100, 200, 400, 800]
    print("=" * 100)
    print("PHASE 36 -- UNSUPERVISED HIERARCHICAL RECALL (discover_categories_v2 "
          "+ distinctness k-selection)")
    print("=" * 100)
    header = (f"{'vocab':>6} {'flat':>7} {'hier_orc':>9} {'hier_dsc':>9} {'oracle':>7} "
              f"{'k*':>3} {'keff':>4} {'purity':>7} {'Vmeas':>6} {'MIz':>6} "
              f"{'cov':>8} {'mem':>5} {'sec':>7}")

    rows = []
    for n_words in sweep:
        t0 = time.time()
        n_words, per_cat, cat, next_cat, emb = build_vocab(n_words)
        seq_len = STEPS_PER_WORD * n_words
        train_seq = sample_stream(n_words, per_cat, cat, next_cat, seq_len, seed=99)

        org, n_mem, slot_to_word, slot_cat_oracle, coverage = train_and_route(
            n_words, cat, emb, train_seq, seed=0)

        flat_gram = flat_recall_grammaticality(org, slot_to_word, cat, next_cat)
        hier_orc = hier_recall(org, n_mem, slot_to_word, slot_cat_oracle,
                               N_CATS, cat, next_cat, seed=1)
        if n_words == sweep[0]:
            # router-generalization check: at n_cats=N_CATS the generalized
            # router must equal phase 35's verbatim
            ref = phase35_hier(org, n_mem, slot_to_word, slot_cat_oracle,
                               cat, next_cat, seed=1)
            assert abs(ref - hier_orc) < 1e-12, (ref, hier_orc)
            print(f"[check] generalized router == phase-35 router at vocab={n_words}: "
                  f"{hier_orc:.3f}\n")

        print(f"  vocab={n_words}: discovery sweep over k (n_mem={n_mem})")
        labels, k_star, k_eff, d_star, mi, p99, z = discover(org)
        purity, vm = purity_and_vmeasure(labels, slot_to_word, slot_cat_oracle)
        slot_cat_disc = {k: int(labels[k]) for k in range(n_mem)}
        hier_dsc = hier_recall(org, n_mem, slot_to_word, slot_cat_disc,
                               k_eff, cat, next_cat, seed=2)

        oracle_seq = sample_stream(n_words, per_cat, cat, next_cat, 1000, seed=200)
        gram_oracle = grammaticality(oracle_seq, cat, next_cat)

        sec = time.time() - t0
        rows.append(dict(vocab=n_words, flat=flat_gram, hier_orc=hier_orc,
                         hier_dsc=hier_dsc, oracle=gram_oracle, k_star=k_star,
                         k_eff=k_eff, purity=purity, vm=vm, z=z, mi=mi, p99=p99,
                         cov=coverage, n_mem=n_mem))
        print(f"    selected k*={k_star} (eff {k_eff}, distinctness {d_star:.4f}); "
              f"certificate MI={mi:.3f} vs null p99={p99:.3f} (z={z:.0f})")
        print(header)
        print(f"{n_words:>6} {flat_gram:>7.3f} {hier_orc:>9.3f} {hier_dsc:>9.3f} "
              f"{gram_oracle:>7.3f} {k_star:>3} {k_eff:>4} {purity:>7.3f} {vm:>6.3f} "
              f"{z:>6.0f} {coverage:>4}/{n_words:<3} {n_mem:>5} {sec:>7.1f}\n")

    print("=" * 100)
    print("VERDICT")
    print("=" * 100)

    # phase-35 reproduction check (oracle arms re-run in-process)
    for v, ref in P35_ANCHORS.items():
        r = next(x for x in rows if x['vocab'] == v)
        d_flat = r['flat'] - ref['flat']
        d_hier = r['hier_orc'] - ref['hier']
        flag = "" if max(abs(d_flat), abs(d_hier)) < 0.005 else "  ** MISMATCH -- investigate"
        print(f"[repro] vocab={v}: flat {r['flat']:.3f} (committed {ref['flat']:.3f}), "
              f"hier_oracle {r['hier_orc']:.3f} (committed {ref['hier']:.3f}){flag}")

    print(f"\n{'vocab':>6} {'hier-flat gap recovery: oracle':>32} {'discovered':>11} "
          f"{'orc-dsc deficit':>16} {'1-purity':>9}")
    for r in rows:
        gap = r['oracle'] - r['flat']
        rec_o = (r['hier_orc'] - r['flat']) / gap if abs(gap) > 1e-9 else float('nan')
        rec_d = (r['hier_dsc'] - r['flat']) / gap if abs(gap) > 1e-9 else float('nan')
        print(f"{r['vocab']:>6} {rec_o:>32.2f} {rec_d:>11.2f} "
              f"{r['hier_orc'] - r['hier_dsc']:>16.3f} {1 - r['purity']:>9.3f}")

    last = rows[-1]
    rho = ((last['hier_dsc'] - last['flat']) / (last['hier_orc'] - last['flat'])
           if abs(last['hier_orc'] - last['flat']) > 1e-9 else float('nan'))
    print(f"\nAt {last['vocab']} words: flat {last['flat']:.3f}, discovered-hierarchical "
          f"{last['hier_dsc']:.3f}, oracle-hierarchical {last['hier_orc']:.3f} "
          f"-> rho = (dsc-flat)/(orc-flat) = {rho:.2f}")

    losses = [r for r in rows if r['vocab'] >= 200 and r['hier_dsc'] < r['flat']]
    if losses:
        floor = max(r['purity'] for r in losses)
        print(f"Purity floor OBSERVED: hierarchy loses to flat at "
              f"{[r['vocab'] for r in losses]} words; max purity among losing rows "
              f"= {floor:.3f} (pre-registered honest negative (b)).")
    else:
        floor_lo = min(r['purity'] for r in rows if r['vocab'] >= 200)
        print(f"No purity floor observed in the tested range: hierarchy never loses "
              f"to flat at >= 200 words down to the lowest measured purity "
              f"{floor_lo:.3f} -- the floor, if any, lies below it.")

    if rho >= 0.7 and not losses:
        print(f"\n-> UN-ORACLED FIX CONFIRMED (pre-registered band rho >= 0.7): "
              f"discover_categories_v2 + distinctness k-selection retains "
              f"{rho*100:.0f}% of the oracle router's gap recovery at "
              f"{last['vocab']} words. Phase 35's oracle caveat is CLOSED: "
              f"hierarchical recall is a general-purpose inference-time fix for "
              f"the Attractor Crowding Collapse, with zero labels in the "
              f"mechanism. Next (ROADMAP row 36): fold in phase 33's two "
              f"product-critical fixes and re-run the phase-33 ladder.")
    elif rho >= 0.3:
        print(f"\n-> PARTIAL (pre-registered band 0.3 <= rho < 0.7): discovery-"
              f"routed hierarchy recovers {rho*100:.0f}% of the oracle recovery. "
              f"Check the deficit column above against (1-purity): prediction (a) "
              f"expects them to track. The caveat narrows but does not close.")
    else:
        print(f"\n-> NEGATIVE (pre-registered): rho = {rho:.2f} -- discovered "
              f"categories are too impure to route stage-B resolution; discovery "
              f"quality, not the router, is now the binding constraint. See the "
              f"purity floor above.")


if __name__ == "__main__":
    main()
