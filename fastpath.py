"""
FASTPATH (engineering track E2) -- Numba JIT port of the mechanism's
sequential hot loops: Organism.perceive (all modes: plain, confirm-gated,
pool, amb), recall, and recall2.

Why this exists: the perceive loop is inherently sequential over the stream
(each frame's routing depends on the state the previous frame left behind),
so it cannot be vectorized across time -- but at phase-23 scale (~5M frames
x ~1.6K slots) the NumPy version spends nearly all its time in the Python
interpreter dispatching tiny array ops, not in arithmetic. JIT-compiling the
loop body removes exactly that overhead. CPU is the right tool here; the
embarrassingly parallel statistics (permutation nulls, all-pairs similarity)
are E4's GPU territory, not this file's.

Correctness contract (per E1, regression_harness.py):
  - control flow is a line-for-line port of organism.py's loops -- same
    branches, same update order, same tie-breaking (argmax = first maximum);
  - float results are NOT bitwise identical to NumPy (BLAS matvec and
    fastmath elementwise loops legitimately reorder reductions); the bar is
    the harness's measured tolerance bands, same as any backend port;
  - recall/recall2 reproduce the NumPy backend's RNG stream exactly: noise
    is drawn from the same Generator in the same order (verified: a block
    standard_normal((c,2,N)) draw equals the sequential per-step draws), so
    trajectory differences come only from float reordering, never from
    different randomness.

State is passed explicitly in/out of the kernels (nothing hides in closure
scope), which is deliberate: it is the same externalized-state shape E3's
serialization needs, and it lets perceive consume generator streams in
bounded-memory chunks (phase 23's 1.6M-frame stream never materializes).

The TransitionGraph/EventBoundary logic (organism.py's logic layer) is
inlined in the perceive kernel -- observe/merge/retire on the raw count
matrix P and the boundary's prev-symbol anchor -- because crossing the
Python boundary per committed transition would put the interpreter right
back in the loop. The four graph ops remain the only ways P changes; they
are just compiled now.
"""

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:          # fastpath silently unavailable; organism.py
    HAVE_NUMBA = False       # falls back to the NumPy loops

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco


PERCEIVE_CHUNK = 32768       # frames per kernel call (bounded memory for
                             # generator streams; ~26 MB at N=50 complex128)
RECALL_CHUNK = 8192          # steps of pre-generated noise per kernel call


# ===================================================================== #
# perceive                                                              #
# ===================================================================== #
@njit(cache=True, fastmath=True)
def _nrm(v, target):
    s = 0.0
    for i in range(v.shape[0]):
        s += v[i].real * v[i].real + v[i].imag * v[i].imag
    return v / (np.sqrt(s) + 1e-9) * target


@njit(cache=True, fastmath=True)
def _perceive_chunk(frames, n_valid, z_io, xi, xic, used, count, P,
                    prov, hits, age, nvis, prev_x, state_i,
                    g_in, dt, eta, recruit, p_decay, confirm, probation,
                    pool, active_bar, s_hat, amb, fuse_bar, omega, norm):
    """One chunk of the perceive loop. All state mutates in place.
    state_i = [last_k, boundary_prev, have_prev_x] (int64).
    xic is the maintained conjugate of xi (kept in sync on every row edit)
    so the per-frame K x N overlap runs as one BLAS matvec."""
    N = xi.shape[1]
    K = xi.shape[0]
    z = z_io.copy()
    last_k = state_i[0]
    bprev = state_i[1]
    have_prev = state_i[2]

    for t in range(n_valid):
        x = _nrm(frames[t], norm)

        if pool and confirm > 0:
            saccade = False
            if have_prev == 1:
                d2 = 0.0
                for i in range(N):
                    dr = x[i].real - prev_x[i].real
                    di = x[i].imag - prev_x[i].imag
                    d2 += dr * dr + di * di
                saccade = np.sqrt(d2) > 0.5 * norm
            if saccade:
                # saccade: z is the finished token's settled state -- the
                # unit of evidence; all recruit/pool decisions live here
                ov = np.dot(xic, z)
                mm = np.abs(ov) / N
                for k2 in range(K):
                    if not used[k2]:
                        mm[k2] = -1.0              # random init is not a match
                cand = np.empty(K, np.bool_)
                any_cand = False
                for k2 in range(K):
                    bar = 0.8 / np.sqrt((1 + s_hat) * (1 + s_hat / max(nvis[k2], 1.0)))
                    cand[k2] = mm[k2] > bar
                    any_cand = any_cand or cand[k2]
                conf = 1.0
                if amb > 0.0 and any_cand:         # ambiguity gate (phase 18)
                    best = -1.0
                    kk = 0
                    for k2 in range(K):
                        v = mm[k2] if cand[k2] else -1.0
                        if v > best:
                            best = v
                            kk = k2
                    if prov[kk]:                   # mature winners absorb freely
                        rest = -1.0
                        for k2 in range(K):
                            if k2 != kk and used[k2] and mm[k2] > rest:
                                rest = mm[k2]
                        conf = min(1.0, max(0.0, (mm[kk] - rest) / amb))
                if any_cand and conf > 0.0:        # strongest accepting slot wins
                    best = -1.0
                    kk = 0
                    for k2 in range(K):
                        v = mm[k2] if cand[k2] else -1.0
                        if v > best:
                            best = v
                            kk = k2
                    o_s = np.vdot(xi[kk], z) / N   # conj(xi[kk]) . z
                    z_al = z * np.exp(-1j * np.arctan2(o_s.imag, o_s.real))
                    nvis[kk] += conf               # accepting evidence = staying alive,
                    age[kk] *= (1.0 - conf)        # in proportion to its confidence
                    wv = max(conf / nvis[kk], eta * conf)  # running mean, plasticity floor
                    xi[kk] = _nrm(xi[kk] + wv * (z_al - xi[kk]), norm)
                    xic[kk] = np.conj(xi[kk])
                    if prov[kk]:                   # visit-quality EMA: sustained
                        hits[kk] += conf * (mm[kk] - hits[kk]) / 3.0
                        if hits[kk] > active_bar and nvis[kk] > confirm:
                            prov[kk] = False       # graduated: stable pooled trace
                    oo = np.abs(np.dot(xic, xi[kk])) / N
                    oo[kk] = 0.0
                    for k2 in range(K):
                        if not used[k2]:
                            oo[k2] = 0.0
                    j = 0
                    best = -1.0
                    for k2 in range(K):
                        if oo[k2] > best:
                            best = oo[k2]
                            j = k2
                    if oo[j] > fuse_bar:           # converged duplicates: fuse
                        # tuple order: (prov[j], nvis[kk]) > (prov[kk], nvis[j])
                        pj = 1 if prov[j] else 0
                        pk = 1 if prov[kk] else 0
                        if pj > pk or (pj == pk and nvis[kk] > nvis[j]):
                            keep = kk
                            drop = j
                        else:
                            keep = j
                            drop = kk
                        w_d = nvis[drop] / (nvis[keep] + nvis[drop])
                        o_kd = np.vdot(xi[keep], xi[drop]) / N
                        al = xi[drop] * np.exp(-1j * np.arctan2(o_kd.imag, o_kd.real))
                        xi[keep] = _nrm(xi[keep] + w_d * (al - xi[keep]), norm)
                        xic[keep] = np.conj(xi[keep])
                        nvis[keep] += nvis[drop]
                        hits[keep] = max(hits[keep], hits[drop])
                        prov[keep] = prov[keep] and prov[drop]
                        count[keep] += count[drop]
                        # graph.merge(keep, drop), statement order preserved
                        for c in range(K):
                            P[keep, c] += P[drop, c]
                        for r in range(K):
                            P[r, keep] += P[r, drop]
                        for c in range(K):
                            P[drop, c] = 0.0
                        for r in range(K):
                            P[r, drop] = 0.0
                        used[drop] = False
                        prov[drop] = False
                        count[drop] = 0.0
                        if bprev == drop:          # boundary.remap
                            bprev = keep
                else:
                    tail = 0.0                     # mm[used & ~prov].max(initial=0.0)
                    for k2 in range(K):
                        if used[k2] and not prov[k2] and mm[k2] > tail:
                            tail = mm[k2]
                    all_used = True
                    f = -1
                    for k2 in range(K):
                        if not used[k2]:
                            all_used = False
                            f = k2
                            break
                    if tail < 0.8 * active_bar and not all_used:
                        xi[f] = _nrm(z, norm)      # novel (not a memory's weak tail)
                        xic[f] = np.conj(xi[f])
                        used[f] = True
                        prov[f] = True
                        hits[f] = 0.0
                        age[f] = 0.0
                        nvis[f] = 1.0
            prev_x[:] = x
            have_prev = 1

        s = 0.0                                    # z += dt*dz, renormalize --
        for i in range(N):                         # fused, in place (the numpy
            zi = z[i]                              # expression allocates ~6
            zi = zi + dt * (1j * omega * zi + g_in * (x[i] - zi))  # temporaries
            z[i] = zi                              # per frame)
            s += zi.real * zi.real + zi.imag * zi.imag
        sc = norm / (np.sqrt(s) + 1e-9)
        for i in range(N):
            z[i] *= sc
        o = np.dot(xic, z)                         # overlaps * N; argmax on the
        k = 0                                      # SQUARED magnitude (same
        best = -1.0                                # winner, no per-slot hypot)
        for k2 in range(K):
            m2 = o[k2].real * o[k2].real + o[k2].imag * o[k2].imag
            if m2 > best:
                best = m2
                k = k2
        mk = np.sqrt(best) / N                     # m[k], the only magnitude
        ok = o[k]                                  # the frame flow reads
        if pool and confirm > 0:
            pass                                   # all updates happen at saccades
        else:
            all_used = True
            f = -1
            for k2 in range(K):
                if not used[k2]:
                    all_used = False
                    f = k2
                    break
            if mk < recruit and not all_used:     # novel -> recruit a free slot
                xi[f] = _nrm(z, norm)
                xic[f] = np.conj(xi[f])
                used[f] = True
                k = f
                if confirm > 0:
                    prov[f] = True
                    hits[f] = 0.0
                    age[f] = 0.0
            else:                                  # familiar -> refine memory
                z_al = z * np.exp(-1j * np.arctan2(ok.imag, ok.real))
                xi[k] = _nrm(xi[k] + eta * (z_al - xi[k]), norm)
                xic[k] = np.conj(xi[k])
                used[k] = True
        if confirm > 0:
            if not pool and prov[k] and mk > 0.6 and k != last_k:
                hits[k] += 1.0                     # a fresh visit, not the same dwell
                if hits[k] >= confirm:
                    prov[k] = False                # graduated: pattern recurs
            any_expired = False
            if pool:
                # use it or lose it: ANY slot that stops accepting evidence
                # is dead structure and is recycled
                for k2 in range(K):
                    if used[k2]:
                        age[k2] += 1.0
                        if age[k2] > probation:
                            any_expired = True
            else:
                for k2 in range(K):
                    if prov[k2]:
                        age[k2] += 1.0
                        if age[k2] > probation:
                            any_expired = True
            if any_expired:                        # recycle
                for k2 in range(K):
                    expired = (used[k2] if pool else prov[k2]) and age[k2] > probation
                    if expired:
                        used[k2] = False
                        count[k2] = 0.0
                        prov[k2] = False
                        if pool:
                            for c in range(K):     # graph.retire
                                P[k2, c] = 0.0
                            for r in range(K):
                                P[r, k2] = 0.0
                            if bprev == k2:        # boundary.invalidate
                                bprev = -1
        if mk > active_bar and used[k]:
            count[k] += 1.0
            if not prov[k]:                        # gate: only confirmed, confident
                if k != bprev and bprev >= 0:      # boundary.commit -> graph.observe
                    if p_decay > 0.0:
                        for r in range(K):
                            for c in range(K):
                                P[r, c] *= (1.0 - p_decay)
                    P[bprev, k] += 1.0
                bprev = k
        last_k = k

    z_io[:] = z
    state_i[0] = last_k
    state_i[1] = bprev
    state_i[2] = have_prev


def perceive_fast(org, stream, g_in=4.0, dt=0.05, eta=0.02, recruit=0.55,
                  p_decay=0.0, confirm=0, probation=6000, pool=False,
                  active_bar=0.6, s_hat=0.0, amb=0.0, fuse_bar=0.7):
    """Drop-in backend for Organism.perceive. Consumes any iterable of
    frames (list, array, or generator) in bounded-memory chunks."""
    N, K = org.N, org.K
    xi = np.ascontiguousarray(org.xi, dtype=np.complex128)
    xic = np.conj(xi).copy()
    P = np.ascontiguousarray(org.graph.P, dtype=np.float64)
    z_io = np.ascontiguousarray(org.z, dtype=np.complex128)
    prov = np.zeros(K, np.bool_)
    hits = np.zeros(K, np.float64)
    age = np.zeros(K, np.float64)
    nvis = np.zeros(K, np.float64)
    prev_x = np.zeros(N, np.complex128)
    state_i = np.array([-1, -1, 0], np.int64)      # last_k, boundary.prev, have_prev

    buf = np.empty((PERCEIVE_CHUNK, N), np.complex128)
    fill = 0
    for x in stream:
        buf[fill] = x
        fill += 1
        if fill == PERCEIVE_CHUNK:
            _perceive_chunk(buf, fill, z_io, xi, xic, org.used, org.count, P,
                            prov, hits, age, nvis, prev_x, state_i,
                            float(g_in), float(dt), float(eta), float(recruit),
                            float(p_decay), int(confirm), float(probation),
                            bool(pool), float(active_bar), float(s_hat),
                            float(amb), float(fuse_bar), float(org.omega),
                            float(org.norm))
            fill = 0
    if fill > 0:
        _perceive_chunk(buf, fill, z_io, xi, xic, org.used, org.count, P,
                        prov, hits, age, nvis, prev_x, state_i,
                        float(g_in), float(dt), float(eta), float(recruit),
                        float(p_decay), int(confirm), float(probation),
                        bool(pool), float(active_bar), float(s_hat),
                        float(amb), float(fuse_bar), float(org.omega),
                        float(org.norm))

    if confirm > 0:                                # purge still-unconfirmed slots
        org.used[prov] = False
        org.count[prov] = 0
    org.xi = xi
    org.graph.P = P
    org.z = z_io


# ===================================================================== #
# recall / recall2                                                      #
# ===================================================================== #
@njit(cache=True, fastmath=True)
def _recall_chunk(Mc, M, Pn, noise_re, noise_im, z_io, h, seq_buf, state_i,
                  dt, tau_h, lam, gamma, g_rec, Dn, beta, omega, norm,
                  topk, commit, debounce):
    """One chunk of recall2's loop (recall == recall2 with topk >= Ku,
    commit=0.5, debounce=1 -- organism.py documents that equivalence).
    state_i = [cur, cand, streak, n_committed]."""
    Ku, N = M.shape
    z = z_io.copy()
    cur = state_i[0]
    cand = state_i[1]
    streak = state_i[2]
    nseq = state_i[3]
    k_eff = min(topk, Ku)
    amp = np.sqrt(2 * Dn * dt) / np.sqrt(2.0)

    score = np.empty(Ku, np.float64)
    w = np.empty(Ku, np.float64)
    taken = np.empty(Ku, np.bool_)

    for t in range(noise_re.shape[0]):
        o = np.dot(Mc, z) / N
        m = np.abs(o)
        for k in range(Ku):
            fat = 1.0 - lam * h[k]
            if fat < 0.0:
                fat = 0.0
            score[k] = m[k] * fat + gamma * Pn[cur, k] * fat
        if k_eff < Ku:
            # top-k lateral inhibition: softmax over the k_eff best scores
            for k in range(Ku):
                taken[k] = False
                w[k] = 0.0
            mx = -np.inf
            for _ in range(k_eff):                 # select the k_eff largest
                b = -np.inf
                bi = -1
                for k in range(Ku):
                    if not taken[k] and score[k] > b:
                        b = score[k]
                        bi = k
                taken[bi] = True
                if b > mx:
                    mx = b
            ssum = 0.0
            for k in range(Ku):
                if taken[k]:
                    w[k] = np.exp(beta * (score[k] - mx))
                    ssum += w[k]
            for k in range(Ku):
                w[k] /= ssum
        else:
            mx = score[0]
            for k in range(1, Ku):
                if score[k] > mx:
                    mx = score[k]
            ssum = 0.0
            for k in range(Ku):
                w[k] = np.exp(beta * (score[k] - mx))
                ssum += w[k]
            for k in range(Ku):
                w[k] /= ssum
        wp = np.empty(Ku, np.complex128)
        for k in range(Ku):
            wp[k] = w[k] * o[k] / (m[k] + 1e-9)    # w * phase
        T = np.dot(wp, M)
        for i in range(N):
            zi = z[i]
            T[i] = zi + dt * (1j * omega * zi + g_rec * (T[i] - zi)) \
                + amp * complex(noise_re[t, i], noise_im[t, i])
        z = _nrm(T, norm)
        for k in range(Ku):
            h[k] = h[k] + dt / tau_h * (m[k] - h[k])
        a = 0
        b = m[0]
        for k in range(1, Ku):
            if m[k] > b:
                b = m[k]
                a = k
        if a != cur and m[a] > commit:
            streak = streak + 1 if a == cand else 1
            cand = a
            if streak >= debounce:
                seq_buf[nseq] = a
                nseq += 1
                cur = a
                streak = 0
        else:
            streak = 0
            cand = -1

    z_io[:] = z
    state_i[0] = cur
    state_i[1] = cand
    state_i[2] = streak
    state_i[3] = nseq


def _recall_driver(org, steps, dt, tau_h, lam, gamma, g_rec, Dn,
                   topk, commit, debounce):
    M = np.ascontiguousarray(org.mem, dtype=np.complex128)
    Mc = np.conj(M).copy()
    Pn = np.ascontiguousarray(org.Pn, dtype=np.float64)
    Ku = M.shape[0]
    # identical RNG consumption order to the NumPy path: init z with two
    # standard_normal(N) draws, then two per step (block draws == sequential
    # draws for numpy Generators; verified)
    z_io = (org.rng.standard_normal(org.N) + 1j * org.rng.standard_normal(org.N))
    z_io = (z_io / (np.linalg.norm(z_io) + 1e-9) * org.norm).astype(np.complex128)
    h = np.zeros(Ku, np.float64)
    seq_buf = np.empty(steps, np.int64)
    state_i = np.array([0, -1, 0, 0], np.int64)    # cur, cand, streak, n_committed
    done = 0
    while done < steps:
        c = min(RECALL_CHUNK, steps - done)
        noise = org.rng.standard_normal((c, 2, org.N))
        _recall_chunk(Mc, M, Pn, np.ascontiguousarray(noise[:, 0, :]),
                      np.ascontiguousarray(noise[:, 1, :]),
                      z_io, h, seq_buf, state_i,
                      float(dt), float(tau_h), float(lam), float(gamma),
                      float(g_rec), float(Dn), float(org.beta),
                      float(org.omega), float(org.norm),
                      int(topk), float(commit), int(debounce))
        done += c
    return seq_buf[:state_i[3]].copy()


def recall_fast(org, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
                g_rec=5.0, Dn=0.004):
    # recall() accepts a hop whenever m[a] > 0.5 and a != cur: that is
    # recall2's rule at commit=0.5, debounce=1, topk disabled
    return _recall_driver(org, steps, dt, tau_h, lam, gamma, g_rec, Dn,
                          topk=org.mem.shape[0], commit=0.5, debounce=1)


def recall2_fast(org, steps=40000, dt=0.05, tau_h=18.0, lam=2.0, gamma=2.5,
                 g_rec=5.0, Dn=0.004, topk=8, commit=0.6, debounce=20):
    return _recall_driver(org, steps, dt, tau_h, lam, gamma, g_rec, Dn,
                          topk=topk, commit=commit, debounce=debounce)


def warmup():
    """Force JIT compilation of all kernels on tiny inputs (compilation is
    cached on disk, so this is one-time per numba/env change)."""
    if not HAVE_NUMBA:
        return
    from organism import Organism
    org = Organism(N=8, K=4, seed=0)
    frames = [org.rng.standard_normal(8) + 1j * org.rng.standard_normal(8)
              for _ in range(30)]
    perceive_fast(org, frames)
    perceive_fast(org, frames, confirm=2, probation=10)
    perceive_fast(org, frames, confirm=2, probation=10, pool=True, amb=0.3)
    org.consolidate(prune_frac=0.0)
    if org.mem.shape[0] > 0:
        recall_fast(org, steps=10)
        recall2_fast(org, steps=10, topk=2)
