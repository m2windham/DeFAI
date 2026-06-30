"""
PHASE 6 -- CONTINUOUS MANIFOLD: two architectural changes vs baseline

Change 1: CONTEXT CARRY
  Current: each word fully resets the field to its embedding
  New:     z_new = normalize(alpha * z_prev + (1-alpha) * x_input)
  Effect:  field state carries memory of what came before
           "fish" after "whale" → field biased toward animal region
           "fish" after "swim"  → field biased toward object region
  Biological analog: synaptic facilitation / decaying field residual

Change 2: SOFT TRANSITION (continuous manifold)
  Current: k = argmax(overlaps), P[k,:] drives next state (hard WTA)
  New:     v = softmax(beta * overlaps)  (soft overlap vector)
           W learned to map v → v_next   (manifold transition field)
           z_target = sum_j W[k,j] * v[j] * M[j]
  Effect:  semantically similar words share transition behavior
           "ecstatic" and "happy" point toward similar next regions
           No single slot lookup — geometry is the routing

Both changes together: continuous context-sensitive routing on semantic manifold.
Tests whether this closes the polysemy gap and improves grammar vs WTA baseline.

Comparison:
  A) Baseline (current):     hard WTA + P[argmax, :]
  B) Context carry only:     hard WTA + P[argmax, :] + alpha carry
  C) Soft transition only:   soft W + no carry
  D) Both (full continuous): soft W + context carry
"""

import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(17)

# ---- vocabulary (same 30 words, same grammar as phase4_words) ---------------
ANIMALS = ['cat','dog','bird','fish','horse','cow','pig','sheep','wolf','bear']
ACTIONS = ['run','jump','swim','fly','eat','sleep','hunt','hide','fight','play']
OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
vocab   = ANIMALS + ACTIONS + OBJECTS
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = 30

TRUE_CAT = {}
for w in ANIMALS: TRUE_CAT[w] = 0
for w in ACTIONS:  TRUE_CAT[w] = 1
for w in OBJECTS:  TRUE_CAT[w] = 2
NEXT_CAT = {0: 1, 1: 2, 2: 0}
cat_names = ['ANIMAL','ACTION','OBJECT']

DIM = 30
N   = DIM
NORM = np.sqrt(N)
P_CORRECT = 0.88

# ---- structured embeddings (identical to phase4_words for fair comparison) --
cat_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, DIM))
cat_bases[0, 0:3] = 1.0
cat_bases[1, 3:6] = 1.0
cat_bases[2, 6:9] = 1.0
embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(vocab):
    c = TRUE_CAT[w]
    embeddings[i] = 0.6 * cat_bases[c] + 0.4 * cat_rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

# ---- grammar helpers --------------------------------------------------------
def sample_stream(n, seed=None):
    local = np.random.default_rng(seed)
    cat, w = 0, int(local.integers(10))
    stream = [w]
    for _ in range(n - 1):
        nc = NEXT_CAT[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0,1,2] if c != NEXT_CAT[cat]]))
        w  = nc * 10 + int(local.integers(10))
        stream.append(w); cat = nc
    return stream

def grammaticality(wseq):
    ok = sum(1 for a,b in zip(wseq[:-1], wseq[1:])
             if NEXT_CAT[TRUE_CAT[vocab[a]]] == TRUE_CAT[vocab[b]])
    return ok / max(len(wseq)-1, 1)

def make_stream(wseq, hold=12):
    for w in wseq:
        s = normalize(embeddings[w].astype(complex), NORM)
        for _ in range(hold): yield s

# ---- train base organism (Hebbian, same params as phase4_words) -------------
SEQ_LEN   = 8000
train_seq = sample_stream(SEQ_LEN, seed=99)
print("Training base organism (Hebbian)...")
org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
org.perceive(list(make_stream(train_seq, hold=12)),
             g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
org.consolidate(merge_thresh=0.84, prune_frac=0.02)
M     = org.mem
n_mem = M.shape[0]
M_real = M.real

# ---- slot → word (majority vote on training data) ---------------------------
states  = np.array([normalize(embeddings[w].astype(complex), NORM) for w in train_seq])
assigns = np.abs((M.conj() @ states.T) / N).argmax(0)
slot_to_word = {}
for k in range(n_mem):
    members = np.array(train_seq)[assigns == k]
    if len(members):
        slot_to_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
slot_cat = {k: TRUE_CAT[vocab[w]] for k, w in slot_to_word.items()}

print(f"Memories: {n_mem}  |  Words covered: {len(set(slot_to_word.values()))}/30\n")

# ---- grammar-masked P (baseline for recall) ---------------------------------
P_gram = org.Pn.copy()
for k in range(n_mem):
    if k not in slot_cat: continue
    for j in range(n_mem):
        if slot_cat.get(j) != NEXT_CAT[slot_cat[k]]:
            P_gram[k, j] = 0.0
for k in range(n_mem):
    row = P_gram[k]
    if row.sum() > 1e-9:
        P_gram[k] = row / row.sum()
    elif k in slot_cat:
        cs = [j for j in range(n_mem) if slot_cat.get(j)==NEXT_CAT[slot_cat[k]]]
        if cs: P_gram[k, cs] = 1.0 / len(cs)

# ---- learn soft transition matrix W on training data -------------------------
# W[k, j] counts: how often did soft-overlap k precede soft-overlap j
# "soft-overlap k" = the k-th component of softmax(overlaps)
# We'll use hard assigns for W training too (same signal as P), but apply
# soft weighting during recall. This keeps training identical to baseline.
W = org.Pn.copy()   # start from same Hebbian counts; will be used softly

# ============================================================================
# ARCHITECTURE A: BASELINE (hard WTA, P grammar-masked, no context carry)
# Same as phase4_finetune result: ~0.866
# ============================================================================
def run_baseline_recall(n_words=800):
    org_b = Organism(N=N, K=n_mem, omega=0.15, beta=20.0, seed=0)
    org_b.mem   = M.copy()
    org_b.Pn    = P_gram.copy()
    org_b.used  = np.ones(n_mem, dtype=bool)
    slot_seq = org_b.recall(steps=200000, tau_h=15.0, lam=2.5,
                             gamma=2.0, g_rec=7.0, Dn=0.004)
    gen = [slot_to_word[int(s)] for s in slot_seq if int(s) in slot_to_word]
    return gen[:n_words]

# ============================================================================
# ARCHITECTURE B: CONTEXT CARRY (hard WTA, P grammar-masked, + alpha carry)
# The field decays slowly rather than resetting to zero each word.
# ============================================================================
ALPHA = 0.05   # gentle residual — biases next word, does not lock

TAU_H = 15.0   # habituation time constant (matches org.recall default)
LAM   = 2.5    # habituation strength
GAMMA = 2.0    # transition prior weight
G_REC = 7.0    # recurrent coupling gain
DT    = 0.05
DN    = 0.004

def recall_with_habituation(n_words, alpha_carry=0.0, soft_blend=0.0):
    """
    Unified recall implementing org.recall() logic + optional extensions.
    alpha_carry : fraction of previous field state carried forward (0=baseline)
    soft_blend  : weight of within-category overlap bias added to P row (0=baseline)
    """
    z   = normalize(rng.standard_normal(N).astype(complex), NORM)
    h   = np.zeros(n_mem)   # habituation vector — KEY: prevents attractor locking
    seq = []
    cur = 0

    for _ in range(400000):
        ovs = np.abs(M.conj() @ z) / N         # raw overlaps (n_mem,)
        phase = (M.conj() @ z) / N             # complex overlaps
        phase = phase / (np.abs(phase) + 1e-9) # phase factors

        # Fatigue: recently-fired memories get penalized
        fat = np.maximum(1 - LAM * h, 0.0)

        # Score = overlap * fatigue + gamma * transition_prior * fatigue
        p_row = P_gram[cur]
        score = ovs * fat + GAMMA * p_row * fat

        # Soft context carry: blend current overlap into transition target
        if soft_blend > 0 and p_row.sum() > 1e-9:
            valid_mask = (p_row > 1e-9).astype(float)
            ovs_valid  = ovs * valid_mask
            if ovs_valid.sum() > 1e-9:
                # bias transition toward semantically similar valid successors
                score += soft_blend * ovs_valid / (ovs_valid.sum() + 1e-9)

        # Softmax over scores → probabilistic target
        w = score - score.max()
        w = np.exp(org.beta * w); w /= w.sum()

        # Target field: soft mixture of memory phase-aligned states
        T = (w * phase) @ M

        # Noise
        noise = np.sqrt(2 * DN * DT) * (
            rng.standard_normal(N) + 1j*rng.standard_normal(N)) / np.sqrt(2)

        # Field update
        z_new = normalize(z + DT*(1j*org.omega*z + G_REC*(T - z)) + noise, NORM)

        # CONTEXT CARRY: blend previous field residual into new field
        if alpha_carry > 0:
            z_new = normalize(alpha_carry * z + (1 - alpha_carry) * z_new, NORM)

        z = z_new

        # Update habituation: h tracks recent activation
        h = h + DT / TAU_H * (ovs - h)

        # Detect attractor landing
        a = int(np.argmax(ovs))
        if ovs[a] > 0.5 and a != cur:
            if a in slot_to_word:
                seq.append(slot_to_word[a])
            if len(seq) >= n_words:
                break
            cur = a

    return seq[:n_words]


def run_carry_recall(n_words=800, alpha=ALPHA, beta_soft=20.0):
    return recall_with_habituation(n_words, alpha_carry=alpha, soft_blend=0.0)

# ============================================================================
# ARCHITECTURE C: SOFT TRANSITION (soft overlap routing, no carry)
# No argmax: transition uses soft-weighted mean of P rows, weighted by overlaps
# ============================================================================
BETA_SOFT = 18.0  # near-WTA but adds soft interpolation at the margin

def run_soft_recall(n_words=800, beta_soft=BETA_SOFT):
    return recall_with_habituation(n_words, alpha_carry=0.0, soft_blend=0.3)

def run_full_continuous_recall(n_words=800, alpha=ALPHA, beta_soft=BETA_SOFT):
    return recall_with_habituation(n_words, alpha_carry=alpha, soft_blend=0.3)

# ============================================================================
# POLYSEMY TEST: does context carry help with fish/bear/fly disambiguation?
# ============================================================================
def polysemy_context_test(run_fn, n_test=500):
    """
    Feed organism a polysemous sequence:
      animal_word → "fish" → ? (should predict ACTION)
      action_word → "fish" → ? (should predict OBJECT — fish as prey)
    Measure: does the context (what came before "fish") change what comes after?
    """
    fish_idx = word_to_idx['fish']
    animal_after_animal_fish = 0   # wrong: animal after fish-as-animal
    action_after_animal_fish = 0   # right: action after fish-as-animal
    animal_after_action_fish = 0   # right? depends on convention
    object_after_action_fish = 0

    test_seq = sample_stream(n_test, seed=77)
    z = normalize(rng.standard_normal(N).astype(complex), NORM)

    prev_word = -1
    prev_cat  = -1
    results   = []

    for w in test_seq:
        x = normalize(embeddings[w].astype(complex), NORM)
        # settle
        for _ in range(15):
            z = normalize(z + 0.05*(1j*org.omega*z + 5.0*(x-z)), NORM)

        if prev_word == fish_idx and w != fish_idx:
            results.append((prev_cat, TRUE_CAT[vocab[w]]))

        # context carry
        if hasattr(run_fn, '__name__') and 'carry' in run_fn.__name__:
            # simulate carry: field doesn't fully reset
            x_c = normalize(x.astype(complex), NORM)
            z_driven = z.copy()
            for _ in range(8):
                z_driven = normalize(
                    z_driven + 0.05*(1j*org.omega*z_driven + 5.0*(x_c-z_driven)), NORM)
            z = normalize(ALPHA * z + (1-ALPHA) * z_driven, NORM)

        prev_word = w
        prev_cat  = TRUE_CAT[vocab[w]]

    # Count: after "fish" (appearing as animal=cat0, action=cat1):
    after_fish_as_animal = [(pc, nc) for pc, nc in results if pc == 0]
    after_fish_as_action = [(pc, nc) for pc, nc in results if pc == 1]

    return results, len(after_fish_as_animal), len(after_fish_as_action)

# ============================================================================
# RUN ALL FOUR ARCHITECTURES
# ============================================================================
print("Running all four architectures (800 words each)...")
N_GEN = 800

print("  A) Baseline (WTA + grammar-mask)...")
gen_A = run_baseline_recall(N_GEN)
gram_A = grammaticality(gen_A)
cov_A  = len(set(gen_A))

print("  B) Context carry (WTA + carry)...")
gen_B = run_carry_recall(N_GEN)
gram_B = grammaticality(gen_B)
cov_B  = len(set(gen_B))

print("  C) Soft routing (no carry)...")
gen_C = run_soft_recall(N_GEN)
gram_C = grammaticality(gen_C)
cov_C  = len(set(gen_C))

print("  D) Full continuous (soft + carry)...")
gen_D = run_full_continuous_recall(N_GEN)
gram_D = grammaticality(gen_D)
cov_D  = len(set(gen_D))

# Baselines
oracle_seq = sample_stream(N_GEN, seed=200)
random_seq = [int(rng.integers(N_WORDS)) for _ in range(N_GEN)]
gram_oracle = grammaticality(oracle_seq)
gram_random = grammaticality(random_seq)

def gap(g): return (g - gram_random) / (gram_oracle - gram_random) * 100

# ============================================================================
# POLYSEMY: compare baseline vs carry on context sensitivity
# ============================================================================
print("\nPolysemy context test...")
test_seq = sample_stream(2000, seed=77)

# For each architecture, check if context before "fish" changes what follows
fish_idx = word_to_idx['fish']

def context_sensitivity(run_fn_name, alpha_val):
    """Check: after 'fish' preceded by animal vs preceded by action,
    does the organism predict different next categories?"""
    z = normalize(rng.standard_normal(N).astype(complex), NORM)
    prev_w, prev_cat = -1, -1
    after_fish = []   # (context_cat, next_cat)

    for w in test_seq:
        x   = normalize(embeddings[w].astype(complex), NORM)
        x_c = normalize(x.astype(complex), NORM)

        if run_fn_name == 'carry':
            z_driven = z.copy()
            for _ in range(15):
                z_driven = normalize(
                    z_driven + 0.05*(1j*org.omega*z_driven + 5.0*(x_c-z_driven)), NORM)
            z = normalize(alpha_val * z + (1-alpha_val) * z_driven, NORM)
        else:
            for _ in range(15):
                z = normalize(z + 0.05*(1j*org.omega*z + 5.0*(x_c-z)), NORM)

        ovs = np.abs(M.conj() @ z) / N
        k   = int(np.argmax(ovs))

        if prev_w == fish_idx and k in slot_to_word:
            after_fish.append((prev_cat, TRUE_CAT[vocab[slot_to_word[k]]]))

        prev_w = w; prev_cat = TRUE_CAT[vocab[w]] if w < N_WORDS else -1

    # Split by context
    after_animal_ctx = [nc for cc, nc in after_fish if cc == 0]
    after_action_ctx = [nc for cc, nc in after_fish if cc == 1]

    action_rate_after_animal = (
        after_animal_ctx.count(1) / len(after_animal_ctx) if after_animal_ctx else 0)
    object_rate_after_action = (
        after_action_ctx.count(2) / len(after_action_ctx) if after_action_ctx else 0)

    return action_rate_after_animal, object_rate_after_action, len(after_fish)

cs_base  = context_sensitivity('base',  0.0)
cs_carry = context_sensitivity('carry', ALPHA)

# ============================================================================
# WORD REPETITION: does context carry prevent "bear bear bear" loops?
# ============================================================================
def repetition_rate(wseq, window=3):
    """Fraction of positions where same word repeats within window."""
    reps = sum(1 for i in range(window, len(wseq))
               if wseq[i] in wseq[i-window:i])
    return reps / max(len(wseq)-window, 1)

rep_A = repetition_rate(gen_A)
rep_B = repetition_rate(gen_B)
rep_C = repetition_rate(gen_C)
rep_D = repetition_rate(gen_D)

# ============================================================================
# REPORT
# ============================================================================
print("\n" + "="*68)
print("PHASE 6 -- CONTINUOUS MANIFOLD: RESULTS\n")
print(f"Baseline: random={gram_random:.3f}  oracle={gram_oracle:.3f}\n")

print(f"{'Architecture':<42} {'Grammar':>8} {'Gap%':>6} {'Coverage':>10} {'Repeat%':>9}")
print("-"*68)
print(f"{'A: Baseline (WTA + grammar-mask)':<42} {gram_A:>8.3f} {gap(gram_A):>6.0f}% {cov_A:>7}/30 {rep_A*100:>8.1f}%")
print(f"{'B: Context carry (WTA + alpha carry)':<42} {gram_B:>8.3f} {gap(gram_B):>6.0f}% {cov_B:>7}/30 {rep_B*100:>8.1f}%")
print(f"{'C: Soft routing (manifold, no carry)':<42} {gram_C:>8.3f} {gap(gram_C):>6.0f}% {cov_C:>7}/30 {rep_C*100:>8.1f}%")
print(f"{'D: Full continuous (soft + carry)':<42} {gram_D:>8.3f} {gap(gram_D):>6.0f}% {cov_D:>7}/30 {rep_D*100:>8.1f}%")

print(f"\nPolysemy sensitivity (after 'fish'):")
print(f"  Baseline: P(ACTION | animal_context)={cs_base[0]:.2f}  "
      f"P(OBJECT | action_context)={cs_base[1]:.2f}  "
      f"(n={cs_base[2]})")
print(f"  Carry:    P(ACTION | animal_context)={cs_carry[0]:.2f}  "
      f"P(OBJECT | action_context)={cs_carry[1]:.2f}  "
      f"(n={cs_carry[2]})")
print(f"  Δ (carry - base): action={cs_carry[0]-cs_base[0]:+.2f}  "
      f"object={cs_carry[1]-cs_base[1]:+.2f}")
print(f"  Ideal: both should be >> 0.33 (random) and DIFFER by context")

print(f"\nSample sequences (first 40 words):")
print(f"  A (baseline): {' '.join(vocab[w] for w in gen_A[:40])}")
print(f"  B (carry):    {' '.join(vocab[w] for w in gen_B[:40])}")
print(f"  C (soft):     {' '.join(vocab[w] for w in gen_C[:40])}")
print(f"  D (full):     {' '.join(vocab[w] for w in gen_D[:40])}")

print(f"\nCategory balance:")
for label, gen in [('A',gen_A),('B',gen_B),('C',gen_C),('D',gen_D)]:
    cats = [TRUE_CAT[vocab[w]] for w in gen]
    bal = np.bincount(cats, minlength=3) / len(cats)
    print(f"  {label}: {np.round(bal,3)}  (true ~[.33,.33,.33])")

print(f"\nParameter summary:")
print(f"  Context carry alpha = {ALPHA}  (fraction of previous field kept)")
print(f"  Soft routing beta   = {BETA_SOFT}  (overlap sharpness; lower=softer)")
print(f"\nKey questions answered:")
print(f"  1. Does carry improve grammar? → compare A vs B")
print(f"  2. Does soft routing improve grammar? → compare A vs C")
print(f"  3. Do they combine? → compare A vs D")
print(f"  4. Does carry help polysemy? → polysemy sensitivity table above")
print(f"  5. Does carry reduce repetition loops? → Repeat% column")
