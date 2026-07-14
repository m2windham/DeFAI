"""
E2: NUMBA BACKEND -- JIT kernels for the sequential hot loops.

The perceive/recall loops are interpreter-bound, not arithmetic-bound: each
frame does a handful of small-vector numpy ops whose Python call overhead
dominates at N=30..128. This module ports them op-for-op into numba kernels.

Contract (engineering track E2, per ROADMAP.md):
  - organism.py stays the REFERENCE implementation; this is an opt-in
    backend: `from organism_numba import NumbaOrganism` or set
    DEFAI_BACKEND=numba when running regression_harness.py.
  - Bitwise equality is NOT the bar (reduction order legitimately differs:
    np.linalg.norm is BLAS, the kernel's norm is a sequential sum). The bar
    is E1: all 23 regression_harness.py checks must pass under this backend.
  - The logic/language architecture is preserved at the API level: the
    kernel updates self.graph.P in place and the class syncs nothing else --
    TransitionGraph/EventBoundary remain the Python-facing interface; the
    kernel is the fused backend of perceive+boundary, an implementation
    detail behind the same seam. recall()'s noise is drawn from self.rng in
    Python (chunked) so the stochastic stream matches the reference
    generator's order.

What is NOT ported, and why (measured, not guessed): consolidate() is
O(K^2) over <=60 slots -- microseconds; PolysemyOrganism.perceive_polysemy
is corpus-tier and joins when its harness checks do.
"""

import numpy as np
from numba import njit

from organism import Organism


@njit(cache=True)
def _norm(v, norm):
    s = 0.0
    for i in range(v.shape[0]):
        s += v[i].real * v[i].real + v[i].imag * v[i].imag
    return v / (np.sqrt(s) + 1e-9) * norm


@njit(cache=True)
def _perceive_kernel(fr, z, xi, used, count, P, prov, hits, age, nvis,
                     N, K, norm, omega, g_in, dt, eta, recruit, p_decay,
                     confirm, probation, pool, active_bar, s_hat, amb,
                     fuse_bar):
    """Op-for-op port of Organism.perceive + the EventBoundary commits.
    Mutates xi/used/count/P (and the pool-state arrays) in place; returns z.
    `prev` is the boundary's previous-symbol anchor."""
    prev = -1
    prev_x_valid = False
    prev_x = np.zeros(N, np.complex128)
    last_k = -1
    for t in range(fr.shape[0]):
        x = _norm(fr[t], norm)
        if pool and confirm > 0:
            if prev_x_valid:
                d = 0.0
                for i in range(N):
                    dr = x[i].real - prev_x[i].real
                    di = x[i].imag - prev_x[i].imag
                    d += dr * dr + di * di
                if np.sqrt(d) > 0.5 * norm:
                    # saccade: settled token; all recruit/pool decisions here
                    mm = np.abs(xi.conj() @ z) / N
                    for kk_ in range(K):
                        if not used[kk_]:
                            mm[kk_] = -1.0
                    cand_any = False
                    kk = -1
                    best = -1.0
                    for kk_ in range(K):
                        bar = 0.8 / np.sqrt((1 + s_hat) * (1 + s_hat / max(nvis[kk_], 1.0)))
                        if mm[kk_] > bar:
                            cand_any = True
                            if mm[kk_] > best:
                                best = mm[kk_]; kk = kk_
                    conf = 1.0
                    if amb > 0 and cand_any and prov[kk]:
                        rest = -1.0
                        for kk_ in range(K):
                            if used[kk_] and kk_ != kk and mm[kk_] > rest:
                                rest = mm[kk_]
                        conf = min(1.0, max(0.0, (mm[kk] - rest) / amb))
                    if cand_any and conf > 0.0:
                        o_s = 0.0 + 0.0j
                        for i in range(N):
                            o_s += np.conj(xi[kk, i]) * z[i]
                        o_s /= N
                        th = np.arctan2(o_s.imag, o_s.real)
                        rot = complex(np.cos(th), -np.sin(th))
                        z_al = z * rot
                        nvis[kk] += conf
                        age[kk] *= (1.0 - conf)
                        wv = max(conf / nvis[kk], eta * conf)
                        xi[kk] = _norm(xi[kk] + wv * (z_al - xi[kk]), norm)
                        if prov[kk]:
                            hits[kk] += conf * (mm[kk] - hits[kk]) / 3.0
                            if hits[kk] > active_bar and nvis[kk] > confirm:
                                prov[kk] = False
                        oo = np.abs(xi.conj() @ xi[kk]) / N
                        oo[kk] = 0.0
                        j = -1; bo = -1.0
                        for kk_ in range(K):
                            if used[kk_] and oo[kk_] > bo:
                                bo = oo[kk_]; j = kk_
                        if j >= 0 and bo > fuse_bar:
                            # lexicographic (prov, nvis) tie-break, as reference
                            if (prov[j] and not prov[kk]) or (prov[j] == prov[kk] and nvis[kk] > nvis[j]):
                                keep, drop = kk, j
                            else:
                                keep, drop = j, kk
                            w_d = nvis[drop] / (nvis[keep] + nvis[drop])
                            o_kd = 0.0 + 0.0j
                            for i in range(N):
                                o_kd += np.conj(xi[keep, i]) * xi[drop, i]
                            o_kd /= N
                            th2 = np.arctan2(o_kd.imag, o_kd.real)
                            rot2 = complex(np.cos(th2), -np.sin(th2))
                            al = xi[drop] * rot2
                            xi[keep] = _norm(xi[keep] + w_d * (al - xi[keep]), norm)
                            nvis[keep] += nvis[drop]
                            hits[keep] = max(hits[keep], hits[drop])
                            prov[keep] = prov[keep] and prov[drop]
                            count[keep] += count[drop]
                            for c in range(K):          # graph.merge(keep, drop)
                                P[keep, c] += P[drop, c]
                            for r in range(K):
                                P[r, keep] += P[r, drop]
                            for c in range(K):
                                P[drop, c] = 0.0
                            for r in range(K):
                                P[r, drop] = 0.0
                            used[drop] = False; prov[drop] = False
                            count[drop] = 0.0
                            if prev == drop:            # boundary.remap
                                prev = keep
                    else:
                        mx = 0.0
                        all_used = True
                        f = -1
                        for kk_ in range(K):
                            if used[kk_] and not prov[kk_] and mm[kk_] > mx:
                                mx = mm[kk_]
                            if not used[kk_] and f < 0:
                                f = kk_
                            if not used[kk_]:
                                all_used = False
                        if mx < 0.8 * active_bar and not all_used:
                            xi[f] = _norm(z.copy(), norm)
                            used[f] = True; prov[f] = True
                            hits[f] = 0.0; age[f] = 0.0; nvis[f] = 1.0
            for i in range(N):
                prev_x[i] = x[i]
            prev_x_valid = True
        dz = 1j * omega * z + g_in * (x - z)
        z = _norm(z + dt * dz, norm)
        o = xi.conj() @ z / N
        m = np.abs(o)
        k = 0
        for kk_ in range(1, K):
            if m[kk_] > m[k]:
                k = kk_
        if pool and confirm > 0:
            pass                                        # all updates at saccades
        else:
            all_used = True
            f = -1
            for kk_ in range(K):
                if not used[kk_]:
                    all_used = False
                    if f < 0:
                        f = kk_
            if m[k] < recruit and not all_used:
                xi[f] = _norm(z.copy(), norm)
                used[f] = True
                k = f
                if confirm > 0:
                    prov[f] = True; hits[f] = 0.0; age[f] = 0.0
            else:
                th = np.arctan2(o[k].imag, o[k].real)
                rot = complex(np.cos(th), -np.sin(th))
                z_al = z * rot
                xi[k] = _norm(xi[k] + eta * (z_al - xi[k]), norm)
                used[k] = True
        if confirm > 0:
            if (not pool) and prov[k] and m[k] > 0.6 and k != last_k:
                hits[k] += 1.0
                if hits[k] >= confirm:
                    prov[k] = False
            any_expired = False
            for kk_ in range(K):
                if pool:
                    if used[kk_]:
                        age[kk_] += 1.0
                else:
                    if prov[kk_]:
                        age[kk_] += 1.0
                cond = used[kk_] if pool else prov[kk_]
                if cond and age[kk_] > probation:
                    any_expired = True
                    used[kk_] = False; count[kk_] = 0.0; prov[kk_] = False
                    if pool:
                        for c in range(K):              # graph.retire
                            P[kk_, c] = 0.0
                        for r in range(K):
                            P[r, kk_] = 0.0
                        if prev == kk_:                 # boundary.invalidate
                            prev = -1
            _ = any_expired
        if m[k] > active_bar and used[k]:
            count[k] += 1.0
            if not prov[k]:                             # boundary.commit(k)
                if k != prev and prev >= 0:
                    if p_decay > 0:
                        for r in range(K):
                            for c in range(K):
                                P[r, c] *= (1.0 - p_decay)
                    P[prev, k] += 1.0
                prev = k
        last_k = k
    if confirm > 0:
        for kk_ in range(K):
            if prov[kk_]:
                used[kk_] = False; count[kk_] = 0.0
    return z


@njit(cache=True)
def _recall_kernel(M, Pn, z, h, noise, omega, beta, dt, tau_h, lam, gamma,
                   g_rec, norm, N, cur0, topk, commit, debounce, seq_out):
    """Shared kernel for recall (topk>=Ku, commit=0.5, debounce=1) and
    recall2. Processes one noise chunk; returns (z, cur, cand, streak, nseq)
    so chunks stitch together with the exact reference rng stream."""
    Ku = M.shape[0]
    cur = cur0
    cand = -1
    streak = 0
    nseq = 0
    k_eff = min(topk, Ku)
    score = np.empty(Ku)
    w = np.empty(Ku)
    for t in range(noise.shape[0]):
        o = M.conj() @ z / N
        m = np.abs(o)
        for i in range(Ku):
            fat = max(1.0 - lam * h[i], 0.0)
            score[i] = m[i] * fat + gamma * Pn[cur, i] * fat
        if k_eff < Ku:
            # top-k lateral inhibition without argpartition: selection sort
            # of the k best indices
            thr = -1e18
            for i in range(Ku):
                w[i] = 0.0
            idx = np.empty(k_eff, np.int64)
            usedf = np.zeros(Ku, np.bool_)
            for kk in range(k_eff):
                bi = -1; bv = -1e18
                for i in range(Ku):
                    if not usedf[i] and score[i] > bv:
                        bv = score[i]; bi = i
                idx[kk] = bi; usedf[bi] = True
            mx = score[idx[0]]
            ssum = 0.0
            for kk in range(k_eff):
                e = np.exp(beta * (score[idx[kk]] - mx))
                w[idx[kk]] = e; ssum += e
            for kk in range(k_eff):
                w[idx[kk]] /= ssum
            _ = thr
        else:
            mx = score[0]
            for i in range(1, Ku):
                if score[i] > mx:
                    mx = score[i]
            ssum = 0.0
            for i in range(Ku):
                w[i] = np.exp(beta * (score[i] - mx)); ssum += w[i]
            for i in range(Ku):
                w[i] /= ssum
        T = np.zeros(N, np.complex128)
        for i in range(Ku):
            ph = o[i] / (m[i] + 1e-9)
            wp = w[i] * ph
            for n_ in range(N):
                T[n_] += wp * M[i, n_]
        z = _norm(z + dt * (1j * omega * z + g_rec * (T - z)) + noise[t], norm)
        for i in range(Ku):
            h[i] = h[i] + dt / tau_h * (m[i] - h[i])
        a = 0
        for i in range(1, Ku):
            if m[i] > m[a]:
                a = i
        if a != cur and m[a] > commit:
            if a == cand:
                streak += 1
            else:
                streak = 1
            cand = a
            if streak >= debounce:
                seq_out[nseq] = a; nseq += 1
                cur = a; streak = 0
        else:
            streak = 0; cand = -1
    return z, cur, cand, streak, nseq


@njit(cache=True)
def _recall_plain_kernel(M, Pn, z, h, noise, omega, beta, dt, tau_h, lam,
                         gamma, g_rec, norm, N, cur0, seq_out):
    """recall(): acceptance is m[a]>0.5 with cur updated on EVERY confident
    frame (not just changes) -- subtly different from recall2's rule, so it
    gets its own faithful kernel."""
    Ku = M.shape[0]
    cur = cur0
    nseq = 0
    score = np.empty(Ku)
    w = np.empty(Ku)
    for t in range(noise.shape[0]):
        o = M.conj() @ z / N
        m = np.abs(o)
        for i in range(Ku):
            fat = max(1.0 - lam * h[i], 0.0)
            score[i] = m[i] * fat + gamma * Pn[cur, i] * fat
        mx = score[0]
        for i in range(1, Ku):
            if score[i] > mx:
                mx = score[i]
        ssum = 0.0
        for i in range(Ku):
            w[i] = np.exp(beta * (score[i] - mx)); ssum += w[i]
        for i in range(Ku):
            w[i] /= ssum
        T = np.zeros(N, np.complex128)
        for i in range(Ku):
            ph = o[i] / (m[i] + 1e-9)
            wp = w[i] * ph
            for n_ in range(N):
                T[n_] += wp * M[i, n_]
        z = _norm(z + dt * (1j * omega * z + g_rec * (T - z)) + noise[t], norm)
        for i in range(Ku):
            h[i] = h[i] + dt / tau_h * (m[i] - h[i])
        a = 0
        for i in range(1, Ku):
            if m[i] > m[a]:
                a = i
        if m[a] > 0.5:
            if a != cur:
                seq_out[nseq] = a; nseq += 1
            cur = a
    return z, cur, nseq


class NumbaOrganism(Organism):
    """Drop-in Organism with JIT hot loops. Same API, same state layout,
    same TransitionGraph/EventBoundary seam (the kernel is the fused
    backend behind it). Validated by regression_harness.py under
    DEFAI_BACKEND=numba."""

    CHUNK = 2000     # recall noise chunk: keeps the reference rng call order
                     # (2 draws per step) without materializing steps x N

    def perceive(self, stream, g_in=4.0, dt=0.05, eta=0.02, recruit=0.55,
                 p_decay=0.0, confirm=0, probation=6000, pool=False,
                 active_bar=0.6, s_hat=0.0, amb=0.0, fuse_bar=0.7):
        fr = np.ascontiguousarray(np.array(list(stream), dtype=np.complex128))
        prov = np.zeros(self.K, np.bool_)
        hits = np.zeros(self.K); age = np.zeros(self.K); nvis = np.zeros(self.K)
        self.z = _perceive_kernel(
            fr, self.z.astype(np.complex128), self.xi, self.used, self.count,
            self.graph.P, prov, hits, age, nvis,
            self.N, self.K, self.norm, self.omega, float(g_in), float(dt),
            float(eta), float(recruit), float(p_decay), int(confirm),
            float(probation), bool(pool), float(active_bar), float(s_hat),
            float(amb), float(fuse_bar))

    def _run_recall(self, steps, dt, tau_h, lam, gamma, g_rec, Dn, topk,
                    commit, debounce, plain):
        M = np.ascontiguousarray(self.mem); Ku = M.shape[0]
        # reference draws one real vector then one imag vector: same order
        z = self.rng.standard_normal(self.N) + 1j * self.rng.standard_normal(self.N)
        z = z / (np.linalg.norm(z) + 1e-9) * self.norm
        h = np.zeros(Ku); cur = 0; cand = -1; streak = 0
        seq = []
        done = 0
        amp = np.sqrt(2 * Dn * dt) / np.sqrt(2)
        while done < steps:
            n = min(self.CHUNK, steps - done)
            noise = np.empty((n, self.N), np.complex128)
            for t in range(n):
                noise[t] = amp * (self.rng.standard_normal(self.N) +
                                  1j * self.rng.standard_normal(self.N))
            seq_out = np.empty(n, np.int64)
            if plain:
                z, cur, nseq = _recall_plain_kernel(
                    M, self.Pn, z, h, noise, self.omega, self.beta, dt,
                    tau_h, lam, gamma, g_rec, self.norm, self.N, cur, seq_out)
            else:
                z, cur, cand, streak, nseq = _recall_kernel(
                    M, self.Pn, z, h, noise, self.omega, self.beta, dt,
                    tau_h, lam, gamma, g_rec, self.norm, self.N, cur, topk,
                    commit, debounce, seq_out)
            seq.extend(seq_out[:nseq].tolist())
            done += n
        return np.array(seq, int)

    def recall(self, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
               g_rec=5.0, Dn=0.004):
        return self._run_recall(steps, dt, tau_h, lam, gamma, g_rec, Dn,
                                topk=10 ** 9, commit=0.5, debounce=1, plain=True)

    def recall2(self, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
                g_rec=5.0, Dn=0.004, topk=8, commit=0.6, debounce=20):
        return self._run_recall(steps, dt, tau_h, lam, gamma, g_rec, Dn,
                                topk=topk, commit=commit, debounce=debounce,
                                plain=False)
