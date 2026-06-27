"""
PHASE 4b -- GENERATION QUALITY TEST (v3: clean rebuild)

Core bug fixed: cluster embeddings were too similar → 5 cycle concepts merged
into 2 memories → recall oscillated between 2 slots.

Fix: near-orthogonal random embeddings in 32D. 20 concepts, all distinct.
Cycle 0→1→2→3→4→0 (p=0.85). Rest: uniform over 15 concepts.

Evaluation:
  - KL(generated bigram || true T_matrix)  [statistical fidelity]
  - Cycle fidelity: how often does 0→1, 1→2, 2→3, 3→4, 4→0?
  - Concept coverage: how many of 20 concepts appear?
"""

import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(7)

N_CONCEPTS = 20
DIM = 32
N = DIM
NORM = np.sqrt(N)

# ---- near-orthogonal embeddings (random in 32D, ~0 cosine between any pair) --
raw = rng.standard_normal((N_CONCEPTS, DIM))
# QR to ensure exact orthogonality for the first 20 (DIM=32 > N_CONCEPTS=20)
Q, _ = np.linalg.qr(raw.T)        # Q is (32, 20), columns orthonormal
embeddings = (Q.T * NORM).astype(float)   # (20, 32), each row has norm NORM
# verify near-orthogonality
gram = (embeddings / NORM) @ (embeddings / NORM).T
print(f"Embedding gram matrix max off-diagonal: {np.abs(gram - np.eye(N_CONCEPTS)).max():.6f}")

# ---- transition matrix: tight cycle in 0-4, uniform rest -------------------
T_true = np.zeros((N_CONCEPTS, N_CONCEPTS))
for i in range(5):
    T_true[i, (i + 1) % 5] = 0.85
    for j in range(5, 20): T_true[i, j] = 0.15 / 15
for i in range(5, 20):
    for j in range(N_CONCEPTS):
        if i != j: T_true[i, j] = 1.0 / 19
T_true /= T_true.sum(1, keepdims=True)

# ---- generate observation sequence -----------------------------------------
SEQ_LEN = 6000
seq = [0]
for _ in range(SEQ_LEN - 1):
    seq.append(int(rng.choice(N_CONCEPTS, p=T_true[seq[-1]])))
seq = np.array(seq)

# ---- train organism via perceive() -----------------------------------------
def make_stream(seq, embeddings, hold=10):
    for c in seq:
        s = normalize(embeddings[c].astype(complex), NORM)
        for _ in range(hold):
            yield s

org = Organism(N=N, K=30, omega=0.15, beta=10.0, seed=0)
print("Training organism...")
org.perceive(
    list(make_stream(seq, embeddings)),
    g_in=5.0, dt=0.05, eta=0.02, recruit=0.55
)
kept = org.consolidate(merge_thresh=0.85, prune_frac=0.02)
M = org.mem
n_mem = M.shape[0]
print(f"Memories formed: {n_mem} (target: {N_CONCEPTS})")

# ---- slot → concept via majority vote on training data ---------------------
states = np.array([normalize(embeddings[c].astype(complex), NORM) for c in seq])
assigns = np.abs((M.conj() @ states.T) / N).argmax(0)

slot_to_concept = {}
for k in range(n_mem):
    members = seq[assigns == k]
    if len(members):
        slot_to_concept[k] = int(np.bincount(members, minlength=N_CONCEPTS).argmax())
concept_to_slot = {v: k for k, v in slot_to_concept.items()}

covered = sorted(set(slot_to_concept.values()))
print(f"Concepts covered by memories: {covered} ({len(covered)}/{N_CONCEPTS})")
print(f"Slot→concept: {slot_to_concept}")

# show learned transition matrix vs true (in concept space)
C_P = np.zeros((N_CONCEPTS, N_CONCEPTS))
for k in range(n_mem):
    if org.Pn[k].sum() > 0:
        cf = slot_to_concept.get(k, -1)
        if cf < 0: continue
        for j in range(n_mem):
            ct = slot_to_concept.get(j, -1)
            if ct >= 0:
                C_P[cf, ct] += org.Pn[k, j]
# normalize rows
C_P /= C_P.sum(1, keepdims=True) + 1e-9

print("\nLearned transition (concept space, cycle 0-4):")
for i in range(5):
    row_str = " ".join(f"{C_P[i,j]:.2f}" for j in range(5))
    true_str = " ".join(f"{T_true[i,j]:.2f}" for j in range(5))
    print(f"  from {i}: learned [{row_str}]  true [{true_str}]")

# ---- FREE GENERATION via recall() ------------------------------------------
print("\nRunning recall (free generation)...")
GEN_STEPS = 80000
slot_seq = org.recall(
    steps=GEN_STEPS,
    tau_h=20.0, lam=2.0, gamma=3.0, g_rec=6.0, Dn=0.003
)
gen_org = [slot_to_concept[int(s)] for s in slot_seq if int(s) in slot_to_concept]
GEN_LEN = min(1000, len(gen_org))
gen_org = gen_org[:GEN_LEN]
print(f"Recall: {len(slot_seq)} slot transitions → {len(gen_org)} valid concept steps")

# ---- baselines -------------------------------------------------------------
def true_generate(steps):
    c = 0; out = [c]
    for _ in range(steps - 1):
        c = int(rng.choice(N_CONCEPTS, p=T_true[c])); out.append(c)
    return out

def random_generate(steps):
    return [int(rng.integers(N_CONCEPTS)) for _ in range(steps)]

gen_true = true_generate(GEN_LEN)
gen_rnd = random_generate(GEN_LEN)

# ---- metrics ---------------------------------------------------------------
def empirical_bigram(s, n=N_CONCEPTS):
    B = np.zeros((n, n))
    for a, b in zip(s[:-1], s[1:]):
        if 0 <= a < n and 0 <= b < n:
            B[a, b] += 1
    row = B.sum(1, keepdims=True)
    B /= row + 1e-9
    return B

def kl(P, Q, eps=1e-9):
    total = 0.0
    for i in range(N_CONCEPTS):
        for j in range(N_CONCEPTS):
            if P[i, j] > eps:
                total += P[i, j] * np.log(P[i, j] / (Q[i, j] + eps))
    return total

def cycle_fidelity(s):
    ok = tot = 0
    for a, b in zip(s[:-1], s[1:]):
        if 0 <= a < 5:
            tot += 1
            if b == (a + 1) % 5: ok += 1
    return ok / tot if tot else 0.0

def coverage(s):
    return len(set(c for c in s if 0 <= c < N_CONCEPTS))

B_org = empirical_bigram(gen_org)
B_true_emp = empirical_bigram(gen_true)
B_rnd = empirical_bigram(gen_rnd)

# ---- report ----------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 4b -- GENERATION QUALITY\n")
print(f"Memories: {n_mem}/30  |  Concepts covered: {len(covered)}/{N_CONCEPTS}")
print(f"Gen length: {GEN_LEN} steps\n")

print("KL divergence from true T (lower = better):")
print(f"  Oracle  (sampled from T_true) : {kl(B_true_emp, T_true):.3f}")
print(f"  Organism (recall / free-run)  : {kl(B_org, T_true):.3f}")
print(f"  Random                        : {kl(B_rnd, T_true):.3f}")

print(f"\nCycle fidelity 0→1→2→3→4→0  (true p=0.85):")
print(f"  Oracle   : {cycle_fidelity(gen_true):.3f}")
print(f"  Organism : {cycle_fidelity(gen_org):.3f}")
print(f"  Random   : {cycle_fidelity(gen_rnd):.3f}")

print(f"\nConcept coverage:")
print(f"  Oracle   : {coverage(gen_true)}/{N_CONCEPTS}")
print(f"  Organism : {coverage(gen_org)}/{N_CONCEPTS}")
print(f"  Random   : {coverage(gen_rnd)}/{N_CONCEPTS}")

print(f"\nFirst 50 generated concepts:")
print(f"  Oracle   : {gen_true[:50]}")
print(f"  Organism : {gen_org[:50]}")
