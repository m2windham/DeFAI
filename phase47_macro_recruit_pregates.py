"""
PHASE 47 (T7.5) -- MACRO-RECRUIT PRE-GATES.

WHAT THIS IS FOR. Before any chunk mechanism is built, three questions have
to be answered on measured data: does a principled chunk-SELECTION rule beat
the obvious ones, does a chunk INVENTORY saturate (and if so where), and is
there anything to chunk at the CATEGORY level -- which is phase 29 / T2.4's
pre-registered "level 2 learns nothing" risk, one level up. An exploratory
sandbox session reported answers to all three. This phase replicates them
under the SOP, with the sandbox's own reported confound fixed first.

WHAT IS BEING MEASURED, EXACTLY. A chunk is a bigram that gets its own
symbol. The measure is a two-part description length in NATS: rewrite the
stream by merging the selected bigrams (non-overlapping, left to right --
the standard BPE-style merge, so decoding is unambiguous given the table),
score the rewritten stream under an add-alpha first-order Markov model, and
add the cost of transmitting the table. GAIN = base NLL - (chunked NLL +
table cost). It is a valid code length for the SAME original stream either
way, so the comparison is honest rather than a token-count artifact.

THE SANDBOX'S OWN CONFOUND, FIXED BEFORE ANYTHING ELSE. The sandbox
reported that its top chunks were UNK-heavy -- UNK was 25.3% of its stream
under OOV->UNK mapping -- and that excluding UNK contexts cut the
word-level gain from +2908 to +748 nats, i.e. ~74% of the headline was a
vocabulary artifact. `text_recipe.Recipe` therefore makes OOV policy an
explicit argument with no default, and BOTH streams are reported here.

TWO METHOD CHANGES FROM THE SANDBOX, BOTH TIGHTENING:
  * SELECTION IS HELD OUT. The sandbox selected chunks and scored gain on
    the same text. Chunk selection is a search over ~10^5 candidates, so
    scoring it in-sample is the selection bias T1.6/T1.8/T1.9 each paid
    for. Here every arm selects on a training half and is scored on a
    DISJOINT held-out half, over 5 folds.
  * CATEGORIES COME FROM THE PROJECT'S OWN METHOD. The sandbox used a
    word-level PPMI k-means proxy. This phase uses
    `discover_categories_v2` on a trained organism -- slot-level, the
    standing method -- so a positive result is about the categories the
    mechanism actually discovers.

======================================================================
PRE-REGISTRATION (committed before the run that tests it)
======================================================================

P1 -- SELECTION RULE. Expected-total-gain (per-token divergence x count)
beats BOTH frequency-only and divergence-only at M=64, on held-out text, on
all 5 folds (no sign flip). Sandbox anchors, indicative only: +2908 /
+1740 / +880 nats at 64 slots, in-sample, on its own stream.
  MUNDANE ACCOUNT (M1): expected-total-gain is dominated by its COUNT
  factor at this M, so it is frequency-only wearing a hat, and the two arms
  will be nearly identical while divergence-only trails. Distinguished by
  reporting the rank overlap between the frequency-only and total-gain
  selections at each M -- if overlap is above ~0.9 the "principled" rule is
  not doing independent work and must not be sold as one.

P2 -- THE NULL. Under a predecessor-permutation null (shuffle the stream,
preserving unigram frequencies and destroying bigram structure) every arm's
held-out gain is <= 0. If any arm stays positive under the null, the
measure is broken and nothing else in the phase counts.

P3 -- THE UNK CONFOUND REPRODUCES. At matched M, the gain measured on the
oov="unk" stream is at least 2x the gain on the oov="drop" stream, and the
UNK symbol appears in a disproportionate share of the selected chunks.
  DECISION RULE: whichever way it lands, oov="drop" is the reported default
  and any headline number carries its OOV policy in the same sentence.

P4 -- INVENTORY SATURATES AND THEN REVERSES. Held-out gain rises with M,
peaks in [64, 256], and the MARGINAL gain of at least one later block is
NEGATIVE once the table cost is paid. This is phase 41's fragmentation
lesson at sequence level, and the curve is what supplies the stopping rule
phase 41 lacked. Sandbox anchors: peak near M=128, marginal -472 nats at
M=256 and -1481 at M=512.
  MUNDANE ACCOUNT (M4): the reversal is purely the TABLE COST -- an
  accounting term the mechanism would not pay in the same form -- rather
  than anything about the language. Distinguished by reporting the curve
  BOTH with and without the table term: if it only reverses with the table
  included, say so plainly and quote the stopping rule as table-dependent.

P5 -- THE CATEGORY LEVEL, WHICH IS T2.4's BLOCKER. The category-stream
census clears its nulls and category-level gain per slot exceeds word-level
gain per slot (UNK-controlled). Sandbox anchors: +1458 nats vs +65 null
(+556 vs +33 with UNK positions dropped), 0/5 sign flips, ~+91 nats/slot
vs +11.7 word-level -- i.e. phase 29's pre-registered "level 2 learns
nothing" did NOT fire on that corpus.
  MUNDANE ACCOUNT (M5): the category-stream gain is a clustering artifact
  -- ANY partition of a Zipfian stream produces apparent second-order
  structure through frequency effects alone. The control has to kill
  exactly that, so TWO nulls are run and they kill different things:
    stream-permutation  destroys sequential structure, preserves unigram
                        frequencies -- kills "any stream looks structured"
    LABEL-permutation   permutes the word->category assignment at FIXED
                        category sizes, preserving stream order and the
                        frequency profile -- kills "any partition of a
                        Zipfian stream looks structured", which is M5
  DECISION RULE: the category arm survives only if it clears BOTH nulls on
  all folds. Clearing the stream null alone is exactly the artifact M5
  names and must be reported as a failure, not a partial success.

P6 -- HONEST NEGATIVE, PRE-REGISTERED. If the category arm does not clear
the label-permutation null, phase 29's "level 2 learns nothing" fires on
this corpus, T2.4 stays blocked on its own honest negative, and no chunk
mechanism should be built at the category level. If the word arm's
UNK-controlled payoff is a negligible fraction of total cross-entropy, say
the fraction rather than the nats -- the sandbox's own word-level number
was ~0.19% of cross-entropy for 64 slots, which is the scope sentence that
must travel with any positive.
"""

import time
from collections import Counter

import numpy as np

import text_recipe as tr
from phase45_settling_depth import slot_labels
from polysemy_organism import PolysemyOrganism

M_GRID = (16, 32, 64, 128, 256, 512)
N_FOLDS = 5
ALPHA = 0.5                     # add-alpha smoothing, as in phase 24's MDL
OMEGA, BETA = 0.15, 10.0


# ---------------------------------------------------------------------------
# the description length
# ---------------------------------------------------------------------------

def markov_nll(seq, V, alpha=ALPHA):
    """Total negative log-likelihood (NATS) of `seq` under an add-alpha
    first-order Markov model fitted to `seq` itself. Both arms are fitted the
    same way, so the comparison is between REPRESENTATIONS, not fits."""
    seq = np.asarray(seq, int)
    if seq.size < 2:
        return 0.0
    C = np.zeros((V, V))
    np.add.at(C, (seq[:-1], seq[1:]), 1.0)
    row = C.sum(1, keepdims=True)
    P = (C + alpha) / (row + alpha * V)
    return float(-np.log(P[seq[:-1], seq[1:]]).sum())


def merge_stream(seq, chunks, V):
    """Rewrite `seq` by merging the selected bigrams, non-overlapping and
    left to right. `chunks` is an ordered list of (a, b); the merged symbol
    for chunk i is V + i. Deterministic, so the table plus the rewritten
    stream is a complete code for the original."""
    if not chunks:
        return np.asarray(seq, int), V
    code = {ab: V + i for i, ab in enumerate(chunks)}
    out = []
    i, n = 0, len(seq)
    while i < n:
        if i + 1 < n:
            c = code.get((int(seq[i]), int(seq[i + 1])))
            if c is not None:
                out.append(c)
                i += 2
                continue
        out.append(int(seq[i]))
        i += 1
    return np.asarray(out, int), V + len(chunks)


def table_cost(n_chunks, V):
    """Nats to transmit the chunk table: two base symbols per entry."""
    return 0.0 if n_chunks == 0 else float(n_chunks * 2 * np.log(V))


def gain(seq, chunks, V, with_table=True):
    """base NLL - (chunked NLL + table). Positive = the chunks pay."""
    base = markov_nll(seq, V)
    merged, V2 = merge_stream(seq, chunks, V)
    cost = table_cost(len(chunks), V) if with_table else 0.0
    return base - (markov_nll(merged, V2) + cost)


# ---------------------------------------------------------------------------
# the three selection rules
# ---------------------------------------------------------------------------

def bigram_stats(seq, V):
    seq = np.asarray(seq, int)
    pairs = Counter(zip(seq[:-1].tolist(), seq[1:].tolist()))
    uni = np.bincount(seq, minlength=V).astype(float)
    n = float(seq.size - 1)
    tot = uni.sum()
    rows = []
    for (a, b), c in pairs.items():
        p_ab = c / n
        p_a, p_b = uni[a] / tot, uni[b] / tot
        pmi = np.log(p_ab / max(p_a * p_b, 1e-300) + 1e-300)
        rows.append((a, b, c, float(pmi)))
    return rows


def select(rows, M, rule):
    """frequency: most frequent. divergence: highest per-token PMI (with a
    minimum count so the rule is not pure noise). total: PMI x count."""
    if rule == "frequency":
        key = lambda r: r[2]
    elif rule == "divergence":
        key = lambda r: (r[3] if r[2] >= 5 else -1e18)
    elif rule == "total":
        key = lambda r: r[3] * r[2]
    else:
        raise ValueError(rule)
    top = sorted(rows, key=key, reverse=True)[:M]
    return [(a, b) for a, b, _, _ in top]


def folds(seq, n=N_FOLDS):
    """n disjoint (train, test) halves -- selection never sees its scorer."""
    seq = np.asarray(seq, int)
    L = len(seq) // n
    out = []
    for i in range(n):
        block = seq[i * L:(i + 1) * L]
        h = len(block) // 2
        out.append((block[:h], block[h:]))
    return out


def overlap(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), 1)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    print("PHASE 47 (T7.5): macro-recruit pre-gates\n")
    text = tr.load_gutenberg8()

    streams = {}
    for oov in ("drop", "unk"):
        R = tr.Recipe(text, **tr.RECIPE_20, oov=oov, verbose=True)
        streams[oov] = R
    R = streams["drop"]

    # ---------------- 1. selection rule, held out -------------------------
    print("\n" + "=" * 70)
    print("(1) SELECTION RULE, SELECTED ON TRAIN AND SCORED ON HELD-OUT -- P1")
    V = len(R.vocab)
    fold_list = folds(R.train_seq)
    print(f"  corpus {len(R.train_seq)} tokens, V={V}, {N_FOLDS} folds of "
          f"{len(fold_list[0][0])} train / {len(fold_list[0][1])} test")
    print(f"  {'M':>5} {'rule':>11} | " + " ".join(f"{'f' + str(i):>9}"
                                                   for i in range(N_FOLDS))
          + f" | {'mean':>9} {'min':>9} | vs frequency")
    sel_cache = {}
    for M in M_GRID:
        per_rule = {}
        for rule in ("frequency", "divergence", "total"):
            g = []
            for fi, (tr_s, te_s) in enumerate(fold_list):
                ch = select(bigram_stats(tr_s, V), M, rule)
                sel_cache[(M, rule, fi)] = ch
                g.append(gain(te_s, ch, V))
            per_rule[rule] = np.array(g)
        for rule in ("frequency", "divergence", "total"):
            g = per_rule[rule]
            d = g - per_rule["frequency"]
            tag = "" if rule == "frequency" else (
                f" {d.mean():+9.1f} mean, {d.min():+9.1f} min"
                + ("  0/5 sign flips" if np.all(d > 0) else
                   f"  {int(np.sum(d <= 0))}/5 sign flips"))
            print(f"  {M:>5} {rule:>11} | " + " ".join(f"{x:>9.1f}" for x in g)
                  + f" | {g.mean():>9.1f} {g.min():>9.1f} |{tag}")
        ov = np.mean([overlap(sel_cache[(M, 'total', i)],
                              sel_cache[(M, 'frequency', i)])
                      for i in range(N_FOLDS)])
        verdict = ("the two rules are near-identical -- the principled rule "
                   "is frequency wearing a hat" if ov > 0.9
                   else "the rules select differently")
        print(f"  {'':>5} {'M1 check':>11} | total-vs-frequency selection overlap "
              f"{ov:.3f}  ({verdict})")

    M0 = 64
    g_tot = np.array([gain(te, sel_cache[(M0, "total", i)], V)
                      for i, (_, te) in enumerate(fold_list)])
    g_frq = np.array([gain(te, sel_cache[(M0, "frequency", i)], V)
                      for i, (_, te) in enumerate(fold_list)])
    g_div = np.array([gain(te, sel_cache[(M0, "divergence", i)], V)
                      for i, (_, te) in enumerate(fold_list)])
    p1 = np.all(g_tot > g_frq) and np.all(g_tot > g_div)
    print(f"\n  P1 {'HELD' if p1 else 'MISSED'} at M={M0}: total {g_tot.mean():.1f} "
          f"vs frequency {g_frq.mean():.1f} vs divergence {g_div.mean():.1f}")

    # ---------------- 2. the null -----------------------------------------
    print("\n" + "=" * 70)
    print("(2) PREDECESSOR-PERMUTATION NULL -- P2")
    rng = np.random.default_rng(0)
    print(f"  {'rule':>11} | " + " ".join(f"{'f' + str(i):>9}" for i in range(N_FOLDS))
          + f" | {'mean':>9}")
    null_ok = True
    for rule in ("frequency", "divergence", "total"):
        g = []
        for fi, (tr_s, te_s) in enumerate(fold_list):
            tr_p = rng.permutation(tr_s)
            te_p = rng.permutation(te_s)
            ch = select(bigram_stats(tr_p, V), M0, rule)
            g.append(gain(te_p, ch, V))
        g = np.array(g)
        null_ok &= bool(np.all(g <= 0))
        print(f"  {rule:>11} | " + " ".join(f"{x:>9.1f}" for x in g)
              + f" | {g.mean():>9.1f}")
    print(f"  P2 {'HELD (every arm <= 0 under the null)' if null_ok else 'MISSED -- the measure is broken'}")

    # ---------------- 3. the UNK confound ---------------------------------
    print("\n" + "=" * 70)
    print("(3) THE UNK CONFOUND -- P3 (the sandbox's own reported artifact)")
    Ru = streams["unk"]
    Vu, unk = len(Ru.vocab), Ru.unk
    fu = folds(Ru.train_seq)
    unk_share = float(np.mean(np.asarray(Ru.train_seq) == unk))
    g_drop, g_unk, unk_frac = [], [], []
    for i, (tr_s, te_s) in enumerate(fu):
        ch = select(bigram_stats(tr_s, Vu), M0, "total")
        g_unk.append(gain(te_s, ch, Vu))
        unk_frac.append(np.mean([unk in ab for ab in ch]))
    for i, (tr_s, te_s) in enumerate(fold_list):
        g_drop.append(gain(te_s, sel_cache[(M0, "total", i)], V))
    g_drop, g_unk = np.array(g_drop), np.array(g_unk)
    base_unk = np.mean([markov_nll(te, Vu) for _, te in fu])
    base_drop = np.mean([markov_nll(te, V) for _, te in fold_list])
    print(f"  UNK share of the oov='unk' stream: {unk_share:.3f} "
          f"(sandbox reported 0.253)")
    print(f"  held-out gain at M={M0}: oov='unk' {g_unk.mean():>9.1f} nats  vs  "
          f"oov='drop' {g_drop.mean():>9.1f} nats   ratio "
          f"{g_unk.mean() / max(g_drop.mean(), 1e-9):.2f}x")
    print(f"  share of selected chunks containing UNK: {np.mean(unk_frac):.3f}")
    print(f"  as a FRACTION of total cross-entropy: unk "
          f"{g_unk.mean() / base_unk * 100:.3f}%   drop "
          f"{g_drop.mean() / base_drop * 100:.3f}%   <- the scope sentence")
    print(f"  P3 {'HELD' if g_unk.mean() >= 2 * g_drop.mean() else 'MISSED'} "
          f"(predicted >= 2x). Reported default remains oov='drop'.")

    # ---------------- 4. inventory saturation -----------------------------
    print("\n" + "=" * 70)
    print("(4) INVENTORY SATURATION AND REVERSAL -- P4")
    print(f"  {'M':>5} | {'gain (with table)':>19} {'marginal':>10} | "
          f"{'gain (no table)':>17} {'marginal':>10}")
    prev_w = prev_n = 0.0
    curve = {}
    for M in M_GRID:
        gw = np.mean([gain(te, sel_cache[(M, "total", i)], V, with_table=True)
                      for i, (_, te) in enumerate(fold_list)])
        gn = np.mean([gain(te, sel_cache[(M, "total", i)], V, with_table=False)
                      for i, (_, te) in enumerate(fold_list)])
        curve[M] = (gw, gn)
        print(f"  {M:>5} | {gw:>19.1f} {gw - prev_w:>10.1f} | "
              f"{gn:>17.1f} {gn - prev_n:>10.1f}")
        prev_w, prev_n = gw, gn
    peak = max(curve, key=lambda m: curve[m][0])
    marg_w = [curve[M_GRID[i]][0] - curve[M_GRID[i - 1]][0]
              for i in range(1, len(M_GRID))]
    marg_n = [curve[M_GRID[i]][1] - curve[M_GRID[i - 1]][1]
              for i in range(1, len(M_GRID))]
    print(f"  peak at M={peak}; a marginal block is negative WITH the table: "
          f"{any(m < 0 for m in marg_w)}; WITHOUT it: {any(m < 0 for m in marg_n)}")
    print(f"  P4 {'HELD' if (64 <= peak <= 256 and any(m < 0 for m in marg_w)) else 'MISSED'}"
          f"  (M4: the reversal is "
          f"{'table-cost-driven only' if not any(m < 0 for m in marg_n) else 'present even without the table'})")

    # ---------------- 5. the category level -------------------------------
    print("\n" + "=" * 70)
    print("(5) THE CATEGORY LEVEL -- P5/P6 (T2.4's blocker)")
    K_TXT = min(1200, V * 4)
    org = PolysemyOrganism(N=R.N, K=K_TXT, omega=OMEGA, beta=BETA, seed=0)
    org.perceive(R.stream(tr.HOLD_19), **tr.PERCEIVE_19)
    org.consolidate(merge_thresh=0.84, prune_frac=0.001)
    n_mem = org.mem.shape[0]
    states = R.embeddings[np.asarray(R.train_seq)]
    assign = np.abs((org.mem.conj() @ states.T) / R.N).argmax(0)
    org.discover_categories_v2(verbose=False)
    lab = slot_labels(org, n_mem)
    k = int(lab.max()) + 1
    cat_seq = lab[assign]
    sizes = np.bincount(lab, minlength=k)
    print(f"  organism: {n_mem} memories, k={k} categories "
          f"(discover_categories_v2, slot-level), sizes {sizes.tolist()}")
    print(f"  category stream {len(cat_seq)} tokens, {k * k} candidate bigrams")

    cf = folds(cat_seq)
    g_cat = np.array([gain(te, select(bigram_stats(tr_s, k), min(M0, k * k), "total"), k)
                      for tr_s, te in cf])
    rngc = np.random.default_rng(1)
    g_perm = np.array([gain(rngc.permutation(te),
                            select(bigram_stats(rngc.permutation(tr_s), k),
                                   min(M0, k * k), "total"), k)
                       for tr_s, te in cf])
    g_lab = []
    for tr_s, te in cf:
        perm_lab = rngc.permutation(lab)          # fixed sizes, same stream
        cs = perm_lab[assign]
        cf2 = folds(cs)
        a, b = cf2[0]
        g_lab.append(gain(b, select(bigram_stats(a, k), min(M0, k * k), "total"), k))
    g_lab = np.array(g_lab)
    base_cat = np.mean([markov_nll(te, k) for _, te in cf])
    print(f"  {'arm':>22} | " + " ".join(f"{'f' + str(i):>9}" for i in range(N_FOLDS))
          + f" | {'mean':>9}")
    for name, g in (("category (measured)", g_cat), ("stream-permutation null", g_perm),
                    ("LABEL-permutation null", g_lab)):
        print(f"  {name:>22} | " + " ".join(f"{x:>9.1f}" for x in g)
              + f" | {g.mean():>9.1f}")
    clears_stream = bool(np.all(g_cat > g_perm))
    clears_label = bool(np.all(g_cat > g_lab))
    n_slots = min(M0, k * k)
    print(f"  per-slot: category {g_cat.mean() / max(n_slots, 1):.2f} nats/slot "
          f"vs word-level (oov='drop') {g_drop.mean() / M0:.2f} nats/slot")
    print(f"  as a FRACTION of the category stream's cross-entropy: "
          f"{g_cat.mean() / base_cat * 100:.3f}%")
    print(f"  clears the stream-permutation null on all folds: {clears_stream}")
    print(f"  clears the LABEL-permutation null on all folds:  {clears_label}"
          f"   <- M5's control, the one that matters")
    if clears_stream and clears_label:
        print("  P5 HELD -- T2.4 (phase 29) is unblocked at the CATEGORY level")
    elif clears_stream:
        print("  P5 MISSED / P6 FIRES -- clears the stream null only, which is "
              "exactly the clustering artifact M5 names. This is a FAILURE, not "
              "a partial success; T2.4 stays blocked on its own honest negative")
    else:
        print("  P5 MISSED / P6 FIRES -- phase 29's 'level 2 learns nothing' "
              "fires on this corpus; T2.4 stays blocked")

    print(f"\nTOTAL {time.time() - t_all:.1f}s")
