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
    def perceive(self, stream, g_in=4.0, dt=0.05, eta=0.02, recruit=0.55, p_decay=0.0,
                 confirm=0, probation=6000, pool=False, active_bar=0.6, s_hat=0.0):
        """p_decay: exponential forgetting of transition counts, applied once
        per observed transition (synaptic decay). 0.0 = original behavior
        (counts accumulate forever); 1/p_decay is the effective memory in
        transitions. Added in phase 11 to fix concept-drift inertia.

        confirm / probation (phase 14, noise-robust recruitment): confirm > 0
        makes new slots PROVISIONAL -- they must be re-entered on `confirm`
        separate visits (non-contiguous confident matches) within `probation`
        frames to become permanent, else they are recycled. Provisional slots
        are excluded from transition learning, so one-off noise states can
        neither persist as memories nor contaminate P. Rationale: real
        patterns recur, i.i.d. noise does not. confirm=0 = original behavior.
        Biology: synaptic tagging -- traces need reactivation to consolidate.

        pool / active_bar (phase 17, recruitment beyond sigma*): beyond
        sigma* the phase-14 gate fails analytically -- a genuine revisit's
        overlap 1/sqrt(1+sigma^2 N) sits below the hard 0.6 confirmation
        bar, so evidence must be pooled across occurrences BEFORE the
        keep/discard decision. pool=True makes provisional slots EVIDENCE
        POOLS and defers every recruit/keep decision to SETTLED evidence:

          - an input saccade (a jump in the incoming frame) marks the end
            of a token; the field state at that moment -- fully settled on
            the finished token -- is ONE unit of evidence. Mid-settle
            states never recruit: below sigma* they are the junk source
            (phase 14's mid-transition slots), and with the lowered bar
            they would spawn an unmergeable duplicate on nearly every
            token;
          - each slot accepts evidence at an ANNEALED bar matched to
            what its own evidence should look like at its size: 80% of
            the expected same-pattern overlap 1/sqrt((1+s)(1+s/n)) with
            s = s_hat (the noise energy sigma^2 N) and n the slot's
            pooled visits. At n=1 that is the raw token-token overlap
            (a young pool must bootstrap loosely); as the pool denoises
            it tightens toward the memory-grade bar, quenching the
            contamination that loose matching admits. This matters
            because greedy routing against MANY young pools is
            fluctuation-dominated -- the best of ~40 wrong matches
            rivals the one right match, and static bars leave every
            pool a mixture. Evidence goes to the strongest slot that
            clears its own bar. If none does, it recruits a new pool --
            UNLESS a confirmed memory matches it inside that memory's
            dead zone (80% of `active_bar`): such evidence is probably
            the memory's own weak tail, and routing it into a fresh
            pool hands that pool a selection-biased evidence stream.
            (A wider, spread-scaled zone kills duplicates outright but
            locks late words out of recruitment entirely: with many
            confirmed neighbors, the MAX zone fluctuation blocks every
            attempt, and each uncovered word poisons P with skip
            transitions. The narrow zone + slot lifecycle below is the
            working balance.) Young pools cast no dead zone -- shadowing
            novel evidence with half-formed structure starves real
            words of their pools. The match is judged against the slot
            BEFORE the update (held-out: judging a pool against
            evidence it already absorbed is self-fulfilling and lets
            junk confirm itself);
          - accepted evidence updates the slot by running mean with a
            plasticity floor, weight max(1/n, eta): early visits average
            (the trace's noise shrinks ~1/n while the keep/discard
            decision is still open), mature memories keep adapting on an
            ~1/eta-token window. Per-frame refinement is disabled -- at
            high sigma its few-token effective window re-noisifies every
            memory it touches, which is what kept duplicates unmergeable
            and matches flaky below sigma*;
          - a pool graduates when its VISIT QUALITY -- an EMA (1/3) of
            the held-out match quality of its accepted visits -- exceeds
            `active_bar`, with at least `confirm` visits. A young pool's
            per-visit quality is the raw token-token overlap, whose
            distribution has a fat tail: a simple per-visit threshold
            graduates half-denoised duplicates on a few lucky draws.
            Only genuine denoising lifts the SUSTAINED average past the
            bar, so graduation certifies pooled-trace stability: real
            words rise to it, one-off states, mixtures, and evidence-
            starved duplicates never do;
          - slots FUSE online when they converge: noisy assignment
            inevitably splits a word's evidence across duplicate pools,
            but duplicates of the same pattern approach each other as
            they denoise (the plasticity floor re-centers a maturing
            pool on its recent, bar-filtered, nearly-pure evidence, so
            early contamination washes out), while different patterns
            never do -- so when two slots overlap above 0.7 (higher
            than any plausible different-pattern overlap) they merge,
            pooling their evidence, counts, and transitions. Splitting
            is thereby harmless instead of fatal: the halves find each
            other;
          - USE IT OR LOSE IT: any slot -- provisional or confirmed --
            that goes a full probation window without accepting a
            single visit is recycled, and its transition rows cleared.
            This is the lifecycle answer to the one duplicate that
            survives everything else: a contaminated pool that
            graduates and then FREEZES (its annealed bar rises past
            what its mixed content can attract, so it accepts nothing,
            forever) while still soaking up activity counts. A real
            memory keeps resonating with the stream that made it; dead
            structure fades and frees its slot.

        active_bar is also the confidence bar for counting and transition
        learning (0.6 = original); beyond sigma* it must sit below the
        token-vs-memory overlap or P never learns. pool=False with
        active_bar=0.6 reproduces phase-14 behavior exactly."""
        z = self.z
        prev_active = -1
        prov = np.zeros(self.K, bool)
        hits = np.zeros(self.K); age = np.zeros(self.K); nvis = np.zeros(self.K)
        prev_x = None
        last_k = -1
        for x in stream:
            x = normalize(x, self.norm)
            if pool and confirm > 0:
                if prev_x is not None and np.linalg.norm(x - prev_x) > 0.5 * self.norm:
                    # saccade: z is the finished token's settled state -- the
                    # unit of evidence; all recruit/pool decisions live here
                    mm = np.abs(self.overlaps(z, self.xi))
                    mm[~self.used] = -1.0                  # random init is not a match
                    bars = 0.8 / np.sqrt((1 + s_hat) * (1 + s_hat / np.maximum(nvis, 1.0)))
                    cand = mm > bars
                    if cand.any():                         # strongest accepting slot wins
                        kk = int(np.argmax(np.where(cand, mm, -1.0)))
                        o_s = (self.xi[kk].conj() @ z) / self.N
                        z_al = z * np.exp(-1j * np.angle(o_s))
                        nvis[kk] += 1; age[kk] = 0         # accepting evidence = staying alive
                        wv = max(1.0 / nvis[kk], eta)      # running mean, plasticity floor
                        self.xi[kk] = normalize(
                            self.xi[kk] + wv * (z_al - self.xi[kk]), self.norm)
                        if prov[kk]:                       # visit-quality EMA: sustained
                            hits[kk] += (mm[kk] - hits[kk]) / 3.0  # confidence, not lucky draws
                            if hits[kk] > active_bar and nvis[kk] > confirm:
                                prov[kk] = False           # graduated: stable pooled trace
                        oo = np.abs(self.overlaps(self.xi[kk], self.xi))
                        oo[kk] = 0.0; oo[~self.used] = 0.0
                        j = int(np.argmax(oo))
                        if oo[j] > 0.7:                    # converged duplicates: fuse
                            keep, drop = (kk, j) if (prov[j], nvis[kk]) > (prov[kk], nvis[j]) else (j, kk)
                            w_d = nvis[drop] / (nvis[keep] + nvis[drop])
                            o_kd = (self.xi[keep].conj() @ self.xi[drop]) / self.N
                            al = self.xi[drop] * np.exp(-1j * np.angle(o_kd))
                            self.xi[keep] = normalize(
                                self.xi[keep] + w_d * (al - self.xi[keep]), self.norm)
                            nvis[keep] += nvis[drop]
                            hits[keep] = max(hits[keep], hits[drop])
                            prov[keep] = prov[keep] and prov[drop]
                            self.count[keep] += self.count[drop]
                            self.P[keep] += self.P[drop]; self.P[:, keep] += self.P[:, drop]
                            self.used[drop] = False; prov[drop] = False
                            self.count[drop] = 0; self.P[drop] = 0; self.P[:, drop] = 0
                            if prev_active == drop:
                                prev_active = keep
                    elif (mm[self.used & ~prov].max(initial=0.0) < 0.8 * active_bar
                          and not self.used.all()):        # novel (not a memory's weak tail)
                        f = int(np.argmin(self.used.astype(float)))
                        self.xi[f] = normalize(z, self.norm); self.used[f] = True
                        prov[f] = True; hits[f] = 0; age[f] = 0; nvis[f] = 1
                prev_x = x
            dz = 1j * self.omega * z + g_in * (x - z)     # input-driven (no retrieval: avoids collapse)
            z = normalize(z + dt * dz, self.norm)
            o = self.overlaps(z, self.xi); m = np.abs(o)
            k = int(np.argmax(m))
            if pool and confirm > 0:
                pass                                       # all updates happen at saccades
            elif m[k] < recruit and not self.used.all():  # novel -> recruit a free slot
                f = int(np.argmin(self.used.astype(float)))
                self.xi[f] = normalize(z, self.norm); self.used[f] = True; k = f
                if confirm > 0:
                    prov[f] = True; hits[f] = 0; age[f] = 0
            else:                                          # familiar -> refine memory
                z_al = z * np.exp(-1j * np.angle(o[k]))
                self.xi[k] = normalize(self.xi[k] + eta * (z_al - self.xi[k]), self.norm)
                self.used[k] = True
            if confirm > 0:
                if not pool and prov[k] and m[k] > 0.6 and k != last_k:  # a fresh visit, not the same dwell
                    hits[k] += 1
                    if hits[k] >= confirm:
                        prov[k] = False                     # graduated: pattern recurs
                if pool:
                    # use it or lose it: ANY slot that stops accepting
                    # evidence -- an unconfirmed one-off, or a frozen
                    # mixture whose annealed bar outgrew its mixed
                    # content -- is dead structure and is recycled
                    age[self.used] += 1
                    expired = self.used & (age > probation)
                else:
                    age[prov] += 1
                    expired = prov & (age > probation)
                if expired.any():                           # recycle
                    self.used[expired] = False; self.count[expired] = 0
                    prov[expired] = False
                    if pool:
                        self.P[expired] = 0; self.P[:, expired] = 0
                        if prev_active >= 0 and expired[prev_active]:
                            prev_active = -1
            if m[k] > active_bar and self.used[k]:
                self.count[k] += 1
                if not prov[k]:
                    if k != prev_active and prev_active >= 0:
                        if p_decay > 0:
                            self.P *= (1.0 - p_decay)
                        self.P[prev_active, k] += 1        # Hebbian transition learning
                    prev_active = k
            last_k = k
        if confirm > 0:                                    # purge still-unconfirmed slots
            self.used[prov] = False; self.count[prov] = 0
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

    # ---- PHASE 13: recall with lateral inhibition + hop commitment ---------
    def recall2(self, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
                g_rec=5.0, Dn=0.004, topk=8, commit=0.6, debounce=20):
        """Two fixes over recall(), each ablatable:
        - topk: only the k best-scoring memories compete for the field pull
          (lateral inhibition). With many crowded memories, the full softmax
          blends same-category attractors into a blob and the state flickers
          among them; top-k keeps the pull discrete. topk >= K disables.
        - commit/debounce: a transition is recorded only when the new memory
          exceeds `commit` overlap AND stays the argmax for `debounce`
          consecutive steps -- mid-flight states no longer count as hops.
          debounce=1, commit=0.5 reproduces recall()'s acceptance rule."""
        M = self.mem; Ku = M.shape[0]
        z = normalize(self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N), self.norm)
        h = np.zeros(Ku); seq = []; cur = 0
        cand = -1; streak = 0
        k_eff = min(topk, Ku)
        for s in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat = np.maximum(1 - lam * h, 0.0)
            score = m * fat + gamma * self.Pn[cur] * fat
            if k_eff < Ku:
                top = np.argpartition(score, -k_eff)[-k_eff:]
                w = np.zeros(Ku)
                e = np.exp(self.beta * (score[top] - score[top].max()))
                w[top] = e / e.sum()
            else:
                w = np.exp(self.beta * (score - score.max())); w /= w.sum()
            phase = o / (m + 1e-9)
            T = (w * phase) @ M
            noise = np.sqrt(2 * Dn * dt) * (self.rng.standard_normal(self.N) +
                                            1j * self.rng.standard_normal(self.N)) / np.sqrt(2)
            z = normalize(z + dt * (1j * self.omega * z + g_rec * (T - z)) + noise, self.norm)
            h = h + dt / tau_h * (m - h)
            a = int(np.argmax(m))
            if a != cur and m[a] > commit:
                streak = streak + 1 if a == cand else 1
                cand = a
                if streak >= debounce:
                    seq.append(a); cur = a; streak = 0
            else:
                streak = 0; cand = -1
        return np.array(seq)

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
