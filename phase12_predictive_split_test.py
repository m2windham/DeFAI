"""
PHASE 12 -- PREDICTIVE SPLIT TEST: split on consequence, not appearance

User's insight: shown "fish" in isolation, nobody can disambiguate it --
not us, not the organism. Phase 11 tried to split based on how DIFFERENT
the input residual looks under different contexts. But residual variance
exists for EVERY word (since previous-category fluctuates for everyone in
a stochastic grammar) -- so any fixed threshold either splits everything
or nothing. That's why no global threshold worked.

The right test is a Myhill-Nerode-style state-splitting criterion from
formal grammar induction: a word should be split into multiple memory
slots only if doing so actually reduces predictive uncertainty about
WHAT COMES NEXT. If the successor distribution is the same regardless of
context, splitting buys nothing -- the contexts are operationally
equivalent and should share one slot (this is exactly why "cat" should
NOT split even though its residual varies with the noisy preceding word).
If the successor distribution genuinely differs by context, splitting is
informative -- this is exactly the fish/duck/bear case, where ANIMAL-fish
is always followed by an ACTION-category word and ACTION-fish is always
followed by an OBJECT-category word.

This phase tests, on the real corpus and using only OBSERVABLE quantities
(previous word's EMERGENT category from Phase 10, not the true label):
for every word, does conditioning on previous-category reduce entropy of
the next-category distribution? Compare dual-role words against controls.
"""

import numpy as np
from phase8_true_polysemy import (
    vocab, word_to_idx, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, N_WORDS, PURE_ANIMALS, PURE_ACTIONS, OBJECTS,
)
from phase10_emergent_category import word_to_emergent_cat, cat_slots

# map emergent category slot id -> true label, for readability only (not used in the test)
true_cat_lookup = {}
for w in PURE_ANIMALS: true_cat_lookup[w] = ANIMAL
for w in PURE_ACTIONS: true_cat_lookup[w] = ACTION
for w in OBJECTS: true_cat_lookup[w] = OBJECT

emergent_label_name = {}
for cs in cat_slots:
    members_true = [true_cat_lookup[vocab[w]] for w in range(N_WORDS)
                    if word_to_emergent_cat.get(w) == cs and vocab[w] in true_cat_lookup]
    if members_true:
        emergent_label_name[cs] = cat_names[int(np.bincount(members_true, minlength=3).argmax())]
    else:
        emergent_label_name[cs] = f"cat{cs}"

print(f"Emergent category labels: {emergent_label_name}")

def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / (counts.sum() + 1e-12)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def successor_emergent_cat(t):
    """Emergent category of the word at position t+1 (the word AFTER current)."""
    if t + 1 >= len(train_seq):
        return None
    return word_to_emergent_cat.get(train_seq[t+1])

def predecessor_emergent_cat(t):
    if t - 1 < 0:
        return None
    return word_to_emergent_cat.get(train_seq[t-1])

def predictive_split_gain(word_idx):
    """For all occurrences of word_idx, measure entropy of the successor's
    emergent category (a) unconditionally, (b) conditioned on the previous
    word's emergent category. Gain = unconditional_entropy - conditional_entropy
    (information gain, observable quantities only)."""
    occ_t = [t for t, w in enumerate(train_seq) if w == word_idx]
    succ_cats = [successor_emergent_cat(t) for t in occ_t]
    pred_cats = [predecessor_emergent_cat(t) for t in occ_t]
    valid = [(s, p) for s, p in zip(succ_cats, pred_cats) if s is not None and p is not None]
    if len(valid) < 30:
        return None
    succ_cats = [s for s, p in valid]
    pred_cats = [p for s, p in valid]

    n_succ_cats = len(set(succ_cats))
    n_pred_cats = len(set(pred_cats))

    unconditional = entropy(np.bincount(succ_cats, minlength=max(succ_cats)+1))

    # conditional entropy: weighted average of entropy(successor | prev_cat)
    cond_total = 0.0
    n = len(valid)
    for pc in set(pred_cats):
        idxs = [i for i, p in enumerate(pred_cats) if p == pc]
        sub_succ = [succ_cats[i] for i in idxs]
        h = entropy(np.bincount(sub_succ, minlength=max(succ_cats)+1))
        cond_total += (len(idxs)/n) * h

    gain = unconditional - cond_total
    return dict(n=n, unconditional_H=unconditional, conditional_H=cond_total,
                gain=gain, n_pred_contexts=n_pred_cats)

print("\n" + "="*78)
print("PREDICTIVE SPLIT GAIN: does knowing prev-category reduce successor entropy?\n")
print(f"{'Word':<10}{'Role':<10}{'N':>6}{'H(succ)':>10}{'H(succ|prev)':>14}{'Gain (bits)':>13}")
print("-"*78)

dual_gains = []
control_gains = []

for w in DUAL_WORDS:
    r = predictive_split_gain(word_to_idx[w])
    if r:
        print(f"{w:<10}{'DUAL':<10}{r['n']:>6}{r['unconditional_H']:>10.3f}"
              f"{r['conditional_H']:>14.3f}{r['gain']:>13.3f}")
        dual_gains.append(r['gain'])

CONTROL_SAMPLE = PURE_ANIMALS + PURE_ACTIONS + OBJECTS  # all single-role words
for w in CONTROL_SAMPLE:
    r = predictive_split_gain(word_to_idx[w])
    if r:
        control_gains.append(r['gain'])

print(f"\n{'(controls, n=' + str(len(control_gains)) + ' single-role words, summary only)':<78}")
print(f"  control gain: mean={np.mean(control_gains):.3f}  "
      f"median={np.median(control_gains):.3f}  max={np.max(control_gains):.3f}  "
      f"min={np.min(control_gains):.3f}")
print(f"  dual gain:    mean={np.mean(dual_gains):.3f}   "
      f"median={np.median(dual_gains):.3f}   max={np.max(dual_gains):.3f}   "
      f"min={np.min(dual_gains):.3f}")

# show full control distribution for transparency
print("\nFull control word gains:")
for w in CONTROL_SAMPLE:
    r = predictive_split_gain(word_to_idx[w])
    if r:
        print(f"  {w:<10} gain={r['gain']:.3f}")

print("\n" + "="*78)
print("VERDICT:")
margin = min(dual_gains) - max(control_gains)
if margin > 0.05:
    print(f"  CLEAN SEPARATION: min(dual_gain)={min(dual_gains):.3f} > "
          f"max(control_gain)={max(control_gains):.3f}  (margin={margin:.3f} bits)")
    print("  Predictive (Myhill-Nerode-style) splitting criterion correctly")
    print("  distinguishes genuinely polysemous words from single-role words,")
    print("  using only observable corpus statistics (no true-role labels).")
    print("  This is the missing ingredient from Phase 11's residual-only gate.")
else:
    print(f"  NO CLEAN SEPARATION: min(dual_gain)={min(dual_gains):.3f}  "
          f"max(control_gain)={max(control_gains):.3f}  (margin={margin:.3f} bits)")
    print("  Predictive gain alone does not cleanly separate dual-role words")
    print("  from controls either -- would need a different criterion.")
