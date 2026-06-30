"""
ORGANISM (Phase 1 consolidation): perceive -> form memories -> learn the world's
SEQUENTIAL structure -> recall by generating learned trajectories.

Advance over learn_world.py: the world now has ORDER. Regimes follow a structured
transition graph, not random switching. The organism must learn BOTH the memories
(what the regimes are) AND the transition structure (which follows which), then in
recall -- with no input -- GENERATE sequences that follow the learned structure.
That is the jump from wandering memory to imagining plausible trajectories.

Pipeline (one module, clean API):
  Organism.perceive(stream)  -> competitive memory formation + Hebbian transition
                                 learning across time.
  Organism.consolidate()     -> merge duplicate memories, drop unused slots.
  Organism.recall(steps)     -> autonomous sequence generation: fatigue releases the
                                 current memory, transition prior P[cur,:] biases the
                                 next, field locks onto a likely successor.

Honesty: we score the generated sequences against the TRUE transition graph and
against a random-successor BASELINE.
"""

import numpy as np


def normalize(v, norm):
    return v / (np.linalg.norm(v) + 1e-9) * norm


class Organism:
    def __init__(self, N=128, K=8, omega=0.25, beta=12.0, seed=0):
        self.N, self.K, self.omega, self.beta = N, K, omega, beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.array([normalize(self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N),
                                      self.norm) for _ in range(K)])
        self.used = np.zeros(K, bool)
        self.count = np.zeros(K)            # how often each slot is the confident active memory
        self.P = np.zeros((K, K))           # learned transition counts between memories
        self.z = normalize(self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N), self.norm)

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    # ---- PHASE 1: perceive + form memories + learn transitions -------------
    def perceive(self, stream, g_in=4.0, dt=0.05, eta=0.02, recruit=0.55):
        z = self.z
        prev_active = -1
        for x in stream:
            x = normalize(x, self.norm)
            dz = 1j * self.omega * z + g_in * (x - z)     # input-driven (no retrieval: avoids collapse)
            z = normalize(z + dt * dz, self.norm)
            o = self.overlaps(z, self.xi); m = np.abs(o)
            k = int(np.argmax(m))
            if m[k] < recruit and not self.used.all():    # novel -> recruit a free slot
                f = int(np.argmin(self.used.astype(float)))
                self.xi[f] = normalize(z, self.norm); self.used[f] = True; k = f
            else:                                          # familiar -> refine memory
                z_al = z * np.exp(-1j * np.angle(o[k]))
                self.xi[k] = normalize(self.xi[k] + eta * (z_al - self.xi[k]), self.norm)
                self.used[k] = True
            if m[k] > 0.6:
                self.count[k] += 1
                if k != prev_active and prev_active >= 0:
                    self.P[prev_active, k] += 1            # Hebbian transition learning
                prev_active = k
        self.z = z

    # ---- consolidate: merge duplicate memories, drop unused ---------------
    def consolidate(self, merge_thresh=0.8, prune_frac=0.05):
        # prune junk slots (rarely the confident active memory -> recruited mid-transition)
        keepable = self.count > prune_frac * self.count.max()
        idx = list(np.where(self.used & keepable)[0])
        merged = []
        for k in idx:
            dup = next((j for j in merged
                        if abs(self.overlaps(self.xi[k], self.xi[j:j+1])[0]) > merge_thresh), None)
            if dup is None:
                merged.append(k)
            else:                                          # fold transitions into the kept slot
                self.P[dup] += self.P[k]; self.P[:, dup] += self.P[:, k]
        self.mem = self.xi[merged]
        Pm = self.P[np.ix_(merged, merged)]
        self.Pn = Pm / (Pm.sum(1, keepdims=True) + 1e-9)   # row-normalized transition probs
        return merged

    # ---- PHASE 2: recall = generate learned trajectories ------------------
    def recall(self, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
               g_rec=5.0, Dn=0.004):
        M = self.mem; Ku = M.shape[0]
        z = normalize(self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N), self.norm)
        h = np.zeros(Ku); seq = []; cur = 0
        for s in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat = np.maximum(1 - lam * h, 0.0)
            # selection = freshness (fatigue) + learned transition prior from current
            score = m * fat + gamma * self.Pn[cur] * fat
            w = np.exp(self.beta * (score - score.max())); w /= w.sum()
            phase = o / (m + 1e-9)
            T = (w * phase) @ M
            noise = np.sqrt(2 * Dn * dt) * (self.rng.standard_normal(self.N) +
                                            1j * self.rng.standard_normal(self.N)) / np.sqrt(2)
            z = normalize(z + dt * (1j * self.omega * z + g_rec * (T - z)) + noise, self.norm)
            h = h + dt / tau_h * (m - h)
            a = int(np.argmax(m))
            if m[a] > 0.5:
                if a != cur:
                    seq.append(a)
                cur = a
        return np.array(seq)


# ============================== DEMO / EVAL ================================
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, H, K = 128, 4, 8
    NORM = np.sqrt(N)

    # hidden world: H regimes + a STRUCTURED transition graph (mostly a cycle
    # 0->1->2->3->0 with a little branching). Unknown to the organism.
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    Ttrue = np.array([[0.0, 0.8, 0.1, 0.1],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.1, 0.1, 0.0, 0.8],
                      [0.8, 0.1, 0.1, 0.0]])
    def make_stream(n, dwell=60, noise=0.5):
        h = 0; out = []
        for i in range(n):
            if i % dwell == 0 and i > 0:
                h = rng.choice(H, p=Ttrue[h])
            out.append(G[h] + noise * NORM / np.sqrt(N) *
                       (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
        return out

    org = Organism(N=N, K=K, seed=0)
    org.perceive(make_stream(80000))
    kept = org.consolidate()

    print("ORGANISM (Phase 1): memories + learned world structure + recall\n")
    print(f"slots used {org.used.sum()}/{K} -> after consolidate: {len(kept)} memories")

    # (A) memory capture
    cap = [max(np.abs(org.overlaps(G[h], org.mem))) for h in range(H)]
    print(f"\n(A) regime capture (never saw clean patterns): mean {np.mean(cap):.3f}  {[f'{c:.2f}' for c in cap]}")

    # map each kept memory -> true regime
    mem2reg = [int(np.argmax(np.abs(org.overlaps(org.mem[k], G)))) for k in range(org.mem.shape[0])]
    print(f"    memory->regime map: {mem2reg}")

    # (B) learned transition structure vs truth
    print("\n(B) learned transition matrix (rows=from, cols=to), in REGIME order:")
    Preg = np.zeros((H, H))
    for i in range(org.mem.shape[0]):
        for j in range(org.mem.shape[0]):
            Preg[mem2reg[i], mem2reg[j]] += org.Pn[i, j]
    Preg /= (Preg.sum(1, keepdims=True) + 1e-9)
    for h in range(H):
        print(f"    from {h}: learned {np.round(Preg[h],2)}   true {Ttrue[h]}")

    # (C) recall generates sequences matching learned structure (vs random baseline)
    seq = org.recall(60000)
    reg_seq = np.array([mem2reg[s] for s in seq])
    # empirical bigram of recalled regime sequence
    B = np.zeros((H, H))
    for a, b in zip(reg_seq[:-1], reg_seq[1:]):
        if a != b: B[a, b] += 1
    Bn = B / (B.sum(1, keepdims=True) + 1e-9)
    # score: correlation of recalled transitions with true; baseline = uniform
    mask = ~np.eye(H, dtype=bool)
    corr = np.corrcoef(Bn[mask], Ttrue[mask])[0, 1]
    # baseline: shuffle the recalled sequence -> destroys order, keeps regime freqs
    shuf = reg_seq.copy(); rng.shuffle(shuf)
    Bs = np.zeros((H, H))
    for a, b in zip(shuf[:-1], shuf[1:]):
        if a != b: Bs[a, b] += 1
    Bsn = Bs / (Bs.sum(1, keepdims=True) + 1e-9)
    base = np.corrcoef(Bsn[mask], Ttrue[mask])[0, 1]
    print(f"\n(C) recall ({len(seq)} transitions). recalled-vs-true transition corr = {corr:.3f}")
    print(f"    (shuffled-sequence baseline corr = {base:.3f})")
    print(f"    recalled regime sequence (first 40): {reg_seq[:40].tolist()}")

    ok = np.mean(cap) > 0.7 and corr > 0.5
    print("\nverdict:", "ORGANISM learns world STRUCTURE and recalls plausible trajectories"
          if ok else "partial -- inspect stage")
