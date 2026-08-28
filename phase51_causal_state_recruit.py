"""Phase 51 (T8.1) -- Does `recruit` partition on the wrong equivalence relation?

PRE-REGISTERED 2026-08-28, BEFORE ANY RUN. Repo owner session
(claude/repo-owner-26nu39). Commit this docstring before the run that tests it.

=============================================================================
THE QUESTION
=============================================================================
`Organism.perceive` forms states by APPEARANCE: `recruit` is a similarity
floor on |overlap|, so two inputs land in one slot iff they look alike.

Computational mechanics (Crutchfield/Shalizi) says the minimal sufficient
statistic of the past for predicting the future is the CAUSAL STATE
partition: pasts are equivalent iff they induce the same conditional
distribution over futures. It is the coarsest partition retaining full
predictive power, and it is optimal, minimal and unique.

This project already discovered that criterion. N1 -- its strongest and
most-replicated claim -- says do not ask whether a word's representation
drifts, ask whether knowing the context changes what you predict next.
That is the Myhill-Nerode criterion, and causal states are exactly its
generalization from deterministic automata to stochastic processes.

So N1 is the right equivalence relation, used as a DETECTOR, bolted on
after states were already formed on the wrong one. This phase asks whether
it belongs in `recruit` instead.

Protocol-level ONLY. organism.py is not touched. This is a measurement of
partitions, not a new mechanism.

=============================================================================
THE STREAM, AND WHY IT IS BUILT THIS WAY
=============================================================================
Appearance-equivalence and predictive-equivalence only differ when the data
makes them differ, so the generator constructs both ways they can come
apart, and nothing else:

  ALIASING   (appearance too COARSE): hidden states s1 and s2 emit the SAME
             appearance A1 but have DIFFERENT futures. Disambiguated only by
             the previous symbol. An appearance partition must merge them and
             lose predictive information. This is polysemy.

  SYNONYMY   (appearance too FINE): hidden states s3 and s4 emit DIFFERENT
             appearances A3, A4 but have IDENTICAL futures. An appearance
             partition splits them and wastes a state without losing
             information.

The design is chosen so both partitions have the SAME CARDINALITY (5), which
removes the confound that killed earlier phases: more states trivially
predict better (T1.5), so a cardinality-matched comparison is mandatory.

SCOPE SENTENCE, and it travels with every number this file prints: this is a
CONSTRUCTED stream where the two equivalence relations provably differ. It
measures whether the criterion can be used constructively and what it buys
when it matters. It does NOT establish that real corpora contain enough
aliasing to make it matter. That is the follow-on, not this phase.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  CAUSAL beats APPEARANCE on held-out predictive information at matched
    cardinality, on every paired seed. Predicted margin >= 0.30 bits.
    If this fails, the reframe is wrong and T8.1's re-scope is withdrawn.

P2  The loss is LOCALIZED to the aliased appearance A1. Conditional entropy
    of the next symbol given A1 should be materially higher for APPEARANCE
    than for CAUSAL, while the non-aliased symbols should be near-identical
    (|delta| < 0.05 bits). A diffuse advantage would mean the metric is
    measuring cardinality or estimator bias, not the mechanism.

P3  APPEARANCE does not beat RANDOM-MATCHED by much more than the aliasing
    gap accounts for -- i.e. the appearance partition is a good partition
    that is wrong in one specific, predictable place.

P4  The organism's ACTUAL recruited bank reproduces the appearance partition:
    slot-argmax agreement with ground-truth appearance labels >= 0.95. If
    the organism does not even recover appearance, this phase is measuring
    a broken run and reports that instead.

=============================================================================
M1 -- THE NAMED MUNDANE ACCOUNT (the most boring way this comes out positive)
=============================================================================
The causal partition is FIT on the same successor statistics it is SCORED
on, so it wins by memorizing the training split rather than by capturing
structure. This is the single most likely way for a positive result here to
be fake, and phase 41/33h were both burned by selection.

CONTROL, built in from the start rather than added if the result looks good:
every partition is FIT on a train split and SCORED on a DISJOINT held-out
split. Reported per seed. If the advantage vanishes out of sample, M1 has
fired and the result is negative -- and it will be reported that way.

=============================================================================
KILL RULE (pre-named, adjudicated out loud)
=============================================================================
If CAUSAL does not beat RANDOM-MATCHED at its own cardinality on held-out
data, sign-consistent across all five paired seeds, then the metric or the
estimator is broken and NO conclusion may be drawn in either direction --
the phase is VOID, not positive and not negative. Reported as such.

Statistics: partitions fit on seeds s=0-4 and confirmed on HELD-OUT s=5-9,
paired within seed. At n=5 the smallest attainable two-sided sign-test p is
2/32 = 0.0625, so no within-split result is reported as if it could reach
p<0.05; sign census is the primary evidence.
"""
import os
import numpy as np

RNG_STREAM = 90210
N_DIM = 64            # matches the digits-scale field used by phases 33x
K_SLOTS = 24
HOLD = 6              # frames per emitted symbol
N_SYMBOLS_TRAIN = 6000
N_SYMBOLS_TEST = 6000
NOISE = 0.25

# ---------------------------------------------------------------- generator
# hidden state -> appearance id.  s1 and s2 SHARE appearance 1 (aliasing);
# s3 and s4 have distinct appearances but identical futures (synonymy).
STATE_APPEARANCE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
N_APPEARANCE = 5

# rows are hidden states, entries are P(next hidden state)
T = np.array([
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00],   # s0 -> s1
    [0.00, 0.00, 0.00, 0.50, 0.50, 0.00],   # s1 -> s3/s4     (A1 after A0)
    [0.30, 0.00, 0.00, 0.00, 0.00, 0.70],   # s2 -> s0/s5     (A1 after A4)
    [0.80, 0.00, 0.00, 0.00, 0.00, 0.20],   # s3 -\ identical
    [0.80, 0.00, 0.00, 0.00, 0.00, 0.20],   # s4 -/ futures   (synonyms)
    [0.00, 0.00, 1.00, 0.00, 0.00, 0.00],   # s5 -> s2
])
# ground-truth causal states: {s0},{s1},{s2},{s3,s4},{s5}  -> 5 classes
TRUE_CAUSAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}


def hidden_walk(n, rng):
    s = 0
    out = np.empty(n, dtype=int)
    for i in range(n):
        out[i] = s
        s = rng.choice(6, p=T[s])
    return out


def appearance_codebook(rng):
    """Five quasi-orthogonal complex directions, one per appearance."""
    M = rng.standard_normal((N_APPEARANCE, N_DIM)) + 1j * rng.standard_normal((N_APPEARANCE, N_DIM))
    Q, _ = np.linalg.qr(M.T)
    return (Q.T[:N_APPEARANCE] * np.sqrt(N_DIM))


def emit(states, code, rng):
    """One noisy frame-block per hidden state occurrence."""
    app = np.array([STATE_APPEARANCE[s] for s in states])
    frames = []
    for a in app:
        base = code[a]
        for _ in range(HOLD):
            v = base + NOISE * np.sqrt(N_DIM) / np.sqrt(N_DIM) * (
                rng.standard_normal(N_DIM) + 1j * rng.standard_normal(N_DIM))
            frames.append(v / np.linalg.norm(v) * np.sqrt(N_DIM))
    return frames, app


# ------------------------------------------------------------------ metrics
def _entropy(counts):
    c = np.asarray(counts, dtype=float)
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c / tot
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def predictive_information(labels, nxt, n_next):
    """I(label ; next symbol) in bits, plainly estimated from counts.

    labels: partition label at time t.  nxt: next appearance id at t.
    Both are HELD-OUT observations; the partition was fit elsewhere.
    """
    labels = np.asarray(labels); nxt = np.asarray(nxt)
    H_next = _entropy(np.bincount(nxt, minlength=n_next))
    H_cond, n = 0.0, len(labels)
    for L in np.unique(labels):
        m = labels == L
        H_cond += (m.sum() / n) * _entropy(np.bincount(nxt[m], minlength=n_next))
    return H_next - H_cond, H_cond


def per_symbol_conditional(app, labels, nxt, n_next):
    """H(next | partition) restricted to each APPEARANCE -- for P2's locality
    test. Answers 'where does the appearance partition actually lose?'"""
    out = {}
    for a in np.unique(app):
        m = app == a
        sub_lab, sub_nxt, n = labels[m], nxt[m], m.sum()
        h = 0.0
        for L in np.unique(sub_lab):
            mm = sub_lab == L
            h += (mm.sum() / n) * _entropy(np.bincount(sub_nxt[mm], minlength=n_next))
        out[int(a)] = h
    return out


# ------------------------------------------------------- partition builders
def causal_partition(app, prev_app, nxt, n_next, rng, n_perm=400):
    """N1's criterion used CONSTRUCTIVELY, in both directions.

    SPLIT  an appearance whose next-distribution depends on the previous
           symbol -- this is exactly phase 12's predictive split gain, and
           the threshold is a measured permutation null, not a guess.
    MERGE  appearances whose next-distributions are indistinguishable.

    Returns a function mapping (appearance, prev_appearance) -> class id.
    """
    split_by_prev = {}
    for a in range(N_APPEARANCE):
        m = app == a
        if m.sum() < 100:
            continue
        H_unc = _entropy(np.bincount(nxt[m], minlength=n_next))
        pa, na = prev_app[m], nxt[m]

        def cond_H(pv):
            h, n = 0.0, len(pv)
            for p in np.unique(pv):
                mm = pv == p
                h += (mm.sum() / n) * _entropy(np.bincount(na[mm], minlength=n_next))
            return h

        gain = H_unc - cond_H(pa)
        null = np.array([H_unc - cond_H(rng.permutation(pa)) for _ in range(n_perm)])
        if gain > np.quantile(null, 0.99):          # measured noise floor, not eyeballed
            split_by_prev[a] = True

    # merge appearances with indistinguishable next-distributions
    dists = {}
    for a in range(N_APPEARANCE):
        m = app == a
        if m.sum() == 0:
            continue
        c = np.bincount(nxt[m], minlength=n_next).astype(float)
        dists[a] = c / c.sum()
    groups, assigned = [], set()
    for a in sorted(dists):
        if a in assigned or a in split_by_prev:
            continue
        g = [a]; assigned.add(a)
        for b in sorted(dists):
            if b <= a or b in assigned or b in split_by_prev:
                continue
            if np.abs(dists[a] - dists[b]).sum() < 0.10:   # total-variation
                g.append(b); assigned.add(b)
        groups.append(g)
    merge_map = {a: i for i, g in enumerate(groups) for a in g}
    base = len(groups)

    def assign(a_arr, p_arr):
        out = np.empty(len(a_arr), dtype=int)
        extra = {}
        for i, (a, p) in enumerate(zip(a_arr, p_arr)):
            if a in split_by_prev:
                key = (a, int(p))
                if key not in extra:
                    extra[key] = base + len(extra)
                out[i] = extra[key]
            else:
                out[i] = merge_map.get(a, base + 900)
        return out

    return assign, split_by_prev, len(groups)


def random_matched(app, n_classes, rng):
    """Control: a random partition of the SAME cardinality, so any advantage
    attributable to state count alone shows up here."""
    lut = rng.integers(0, n_classes, size=N_APPEARANCE * 8)
    return lambda a_arr, p_arr: lut[(a_arr * 8 + (p_arr % 8))] % n_classes


# ----------------------------------------------------------------- the run
def run_seed(seed, verbose=False):
    rng = np.random.default_rng(RNG_STREAM + seed)
    code = appearance_codebook(np.random.default_rng(RNG_STREAM + 7000 + seed))

    tr_states = hidden_walk(N_SYMBOLS_TRAIN, rng)
    te_states = hidden_walk(N_SYMBOLS_TEST, rng)
    tr_frames, tr_app = emit(tr_states, code, rng)
    te_frames, te_app = emit(te_states, code, rng)

    def ctx(app_arr):
        return app_arr[1:-1], app_arr[:-2], app_arr[2:]
    tr_a, tr_p, tr_n = ctx(tr_app)
    te_a, te_p, te_n = ctx(te_app)

    # --- the real organism, trained by its real recruitment rule ----------
    from organism import Organism
    org = Organism(N=N_DIM, K=K_SLOTS, seed=seed)
    org.perceive(tr_frames)

    # P4: does the recruited bank recover APPEARANCE at all?
    live = np.flatnonzero(org.used)
    def slot_of(vs):
        o = np.abs(np.asarray(vs) @ org.xi[live].conj().T) / N_DIM
        return live[np.argmax(o, axis=1)]
    probe = np.array([code[a] for a in te_a])
    slots = slot_of(probe)
    agree = 0.0
    for a in range(N_APPEARANCE):
        m = te_a == a
        if m.sum():
            agree += m.sum() * (np.bincount(slots[m]).max() / m.sum())
    agree /= len(te_a)

    # --- the four partitions ---------------------------------------------
    app_part = lambda a_arr, p_arr: a_arr                       # APPEARANCE
    causal_assign, split_set, n_groups = causal_partition(
        tr_a, tr_p, tr_n, N_APPEARANCE, np.random.default_rng(RNG_STREAM + 31 + seed))
    n_causal = len(np.unique(causal_assign(tr_a, tr_p)))
    rnd_part = random_matched(tr_a, n_causal, np.random.default_rng(RNG_STREAM + 55 + seed))
    true_part = lambda a_arr, p_arr: np.array(
        [TRUE_CAUSAL[s] for s in te_states[1:-1]])              # eval-only ORACLE

    res = {}
    for name, fn in (("APPEARANCE", app_part), ("CAUSAL", causal_assign),
                     ("RANDOM-MATCHED", rnd_part), ("TRUE-CAUSAL (oracle)", true_part)):
        lab = fn(te_a, te_p)
        pi, hc = predictive_information(lab, te_n, N_APPEARANCE)
        res[name] = dict(pi=pi, H_cond=hc, n_states=len(np.unique(lab)),
                         per_symbol=per_symbol_conditional(te_a, lab, te_n, N_APPEARANCE))
    res["_agree"] = agree
    res["_split_set"] = sorted(split_set)
    res["_n_groups"] = n_groups
    return res


def main():
    backend = os.environ.get("DEFAI_BACKEND", "auto")
    print("=" * 78)
    print("PHASE 51 (T8.1) -- causal-state vs appearance recruitment")
    print(f"backend={backend}  N={N_DIM} K={K_SLOTS} hold={HOLD} noise={NOISE}")
    print("CONSTRUCTED stream: aliasing (s1/s2 share appearance) + synonymy (s3/s4)")
    print("=" * 78)

    sel, hout = range(0, 5), range(5, 10)
    all_res = {s: run_seed(s) for s in list(sel) + list(hout)}

    def block(seeds, title):
        print(f"\n--- {title} (seeds {list(seeds)}) " + "-" * 28)
        print(f"{'arm':<24}{'states':>7}{'PI (bits)':>12}{'H(next|S)':>12}")
        rows = {}
        for name in ("APPEARANCE", "CAUSAL", "RANDOM-MATCHED", "TRUE-CAUSAL (oracle)"):
            pis = [all_res[s][name]["pi"] for s in seeds]
            ns = [all_res[s][name]["n_states"] for s in seeds]
            hc = [all_res[s][name]["H_cond"] for s in seeds]
            rows[name] = np.array(pis)
            print(f"{name:<24}{np.mean(ns):>7.1f}{np.mean(pis):>12.4f}{np.mean(hc):>12.4f}")
        return rows

    rows_sel = block(sel, "SELECTION")
    rows_out = block(hout, "HELD-OUT")

    print("\n--- P1: CAUSAL - APPEARANCE, paired within seed " + "-" * 22)
    d_out = rows_out["CAUSAL"] - rows_out["APPEARANCE"]
    d_sel = rows_sel["CAUSAL"] - rows_sel["APPEARANCE"]
    print(f"  selection  mean {d_sel.mean():+.4f} bits   positive {int((d_sel>0).sum())}/5")
    print(f"  HELD-OUT   mean {d_out.mean():+.4f} bits   positive {int((d_out>0).sum())}/5"
          f"   range [{d_out.min():+.4f}, {d_out.max():+.4f}]")
    p1 = (d_out > 0).all() and d_out.mean() >= 0.30
    print(f"  P1 (>=0.30 bits, all 5 held-out seeds): {'HELD' if p1 else 'FAILED'}")

    print("\n--- KILL RULE: CAUSAL vs RANDOM-MATCHED (held-out) " + "-" * 19)
    d_r = rows_out["CAUSAL"] - rows_out["RANDOM-MATCHED"]
    print(f"  mean {d_r.mean():+.4f} bits   positive {int((d_r>0).sum())}/5")
    void = not (d_r > 0).all()
    print(f"  {'VOID -- metric broken, no conclusion' if void else 'kill rule NOT triggered'}")

    print("\n--- P2: where does APPEARANCE lose? H(next|S) per appearance " + "-" * 8)
    print(f"{'appearance':<14}{'APPEARANCE':>12}{'CAUSAL':>12}{'delta':>10}   note")
    for a in range(N_APPEARANCE):
        pa = np.mean([all_res[s]["APPEARANCE"]["per_symbol"].get(a, np.nan) for s in hout])
        pc = np.mean([all_res[s]["CAUSAL"]["per_symbol"].get(a, np.nan) for s in hout])
        note = "<-- ALIASED (s1/s2)" if a == 1 else ("synonym pair" if a in (2, 3) else "")
        print(f"A{a:<13}{pa:>12.4f}{pc:>12.4f}{pa-pc:>10.4f}   {note}")

    print("\n--- P4: did the organism recover APPEARANCE at all? " + "-" * 18)
    ag = np.mean([all_res[s]["_agree"] for s in hout])
    print(f"  slot-argmax agreement with appearance labels: {ag:.4f}"
          f"   ({'OK' if ag >= 0.95 else 'BELOW 0.95 -- run is suspect'})")
    print(f"  appearances the criterion chose to SPLIT: "
          f"{[all_res[s]['_split_set'] for s in hout][0]}  (ground truth: [1])")
    print(f"  merge groups formed: {np.mean([all_res[s]['_n_groups'] for s in hout]):.1f}"
          f"   (1-step ground truth: 2 -- see note)")
    print("  NOTE, corrected after the first run: an earlier draft of this line")
    print("  expected 4 groups. That expectation was WRONG. At a 1-step horizon")
    print("  A0 and A4 both go to A1 with probability 1, so they are genuinely")
    print("  synonymous and merging them is correct; A2/A3 merge as designed.")
    print("  Two groups is the right answer and the criterion found it.")

    print("\n" + "=" * 78)
    print("SCOPE: constructed stream where the two equivalence relations differ")
    print("by construction. Measures what the criterion buys WHEN IT MATTERS,")
    print("not that real corpora contain enough aliasing to make it matter.")
    print("=" * 78)


if __name__ == "__main__":
    main()
