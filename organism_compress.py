"""
REPRESENTATION-WIDTH COMPRESSION (T1.8 / phase 33g) -- storage engineering
for the organism's memory store.

Why this exists
---------------
Phase 33d (T1.5) measured the gate's cost branch and localized the failure
precisely: at MATCHED memory count (K=120 slots vs the supervised prototype
bar's 120 prototypes) accuracy nearly matches (0.854 vs 0.872) while the
organism spends 7.8x the bytes. The byte gap is REPRESENTATION WIDTH, not
memory count -- complex128 field memories plus a dense K x K transition
matrix. This module cuts that width.

Three levers, and the honest split between them
-----------------------------------------------
LOSSLESS BY CONSTRUCTION (the state decompresses bit-for-bit):
  * sparse P -- the transition matrix is a COUNT matrix and it is sparse in
    practice (measured on the 33c protocol: 21.6% dense at K=40, 6.1% at
    K=160 -- the dense K^2 term is mostly zeros). CSR storage with a count
    floor of 0 stores exactly the nonzeros and reconstructs P exactly.
  * float32 `count` / `age` -- both hold integer-valued magnitudes far
    below 2**24, where float32 is exact. Guarded: `compress` refuses (and
    keeps float64) if a value would not round-trip exactly.

LOSSY, AND PRICED HERE:
  * complex64 `xi` / `mem` -- halves the dominant term. Attractor capture
    runs at ~0.99 overlaps; float32's ~7 significant digits sit far below
    the decision margins, but "far below" is a prediction, not a fact, so
    phase 33g measures it.
  * count floor > 0 on P -- drops rare transitions. Cheap bytes, real loss
    of graph mass (measured: floor>1 keeps 82% of the mass at K=160).
  * low-rank `xi` factorization -- K slots living in an N-dimensional field
    have rank <= min(K, N), so at K > N the memory bank is exactly
    rank-deficient; truncating below that is lossy.
  * real+phase `xi` layout (T7.1, phase 43) -- store each slot as a REAL
    N-vector plus one phase scalar instead of a complex N-vector. Exact iff
    the slot is a global rotation of a real vector; the loss is exactly the
    residual `real_phase_residual` reports. This is a REPRESENTATION change,
    not a width narrowing: it halves the dominant term again, and it can
    only be taken by a system that does not intend to put content in the
    imaginary channel. Phase 43 measures the residual and prices the fork;
    it does NOT recommend taking it.

What this is NOT
----------------
This is inference/storage engineering. Nothing here re-attributes
occurrences to slots, re-sorts evidence, or re-runs any learning rule:
compression maps a finished state to a narrower encoding of THE SAME
state. Phase 9's closed negative (post-hoc re-attribution inherits the
online mixing it was meant to undo) is about a different operation
entirely and is neither invoked nor re-opened by anything in this file.

Compute width is unchanged. A compressed store is always widened back to
complex128/float64 before it is perceived with -- `Organism.perceive`
enforces this (see its dtype wrapper), so both backends see identical
compute dtypes and the E2 equivalence contract is untouched.

Byte accounting
---------------
`store_bytes(org)` is phase 33d's formula VERBATIM (xi + P + count + age),
so the compressed and uncompressed cost curves are directly comparable.
`CompressedState.nbytes` accounts the same four fields in their compressed
encodings (CSR = data + indices + indptr), nothing hidden.
"""

import numpy as np

WORK_XI = np.complex128        # the compute width, fixed (E2 contract)
WORK_META = np.float64

XI_DTYPES = (np.complex128, np.complex64)
META_DTYPES = (np.float64, np.float32)
_F32_EXACT_MAX = 2.0 ** 24     # integers below this round-trip exactly in float32


def store_bytes(org):
    """Uncompressed store bytes -- phase 33d's `org_state_bytes` formula
    verbatim (xi + P + count + age), measured off the live arrays."""
    return int(org.xi.nbytes + org.P.nbytes + org.count.nbytes + org.age.nbytes)


def real_phase_residual(xi, used=None):
    """Per-slot imaginary energy that survives the BEST global de-rotation.

    For a slot vector v, `min_theta ||Im(e^{-i theta} v)||^2 / ||v||^2`, which
    has the closed form `(1 - |sum_i v_i^2| / sum_i |v_i|^2) / 2` with the
    minimizing angle `angle(sum_i v_i^2) / 2`. It is 0 exactly when v is a
    global rotation of a real vector (the real+phase layout is then exact)
    and 0.5 for an isotropic complex vector (the layout would throw half the
    energy away). Returns the per-slot array, restricted to `used` if given.
    """
    xi = np.asarray(xi, dtype=WORK_XI)
    if used is not None:
        xi = xi[np.asarray(used, dtype=bool)]
    if xi.size == 0:
        return np.zeros(0)
    energy = (np.abs(xi) ** 2).sum(1)
    align = np.abs((xi ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(energy > 0, (1.0 - align / np.maximum(energy, 1e-300)) / 2.0, 0.0)
    return np.clip(r, 0.0, 0.5)


def _f32_safe(a):
    """True when float32 round-trips `a` exactly (integer-valued counts and
    clocks below 2**24 -- the regime every phase script runs in)."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return True
    return bool(np.all(np.abs(a) < _F32_EXACT_MAX)
                and np.array_equal(a.astype(np.float32).astype(np.float64), a))


class CompressionSpec:
    """What to compress and how hard.

    xi_dtype   complex128 (keep) or complex64 (halve the dominant term)
    p_floor    drop P entries with count <= p_floor. 0.0 = LOSSLESS sparse
               (stores exactly the nonzeros); > 0 prunes real graph mass.
    rank       None = no factorization; r = truncate xi to rank r (lossy;
               only ever cheaper than dense when r * (K + N) < K * N)
    meta_dtype float64 or float32 for count/age (float32 is lossless in the
               counting regime and silently declined when it would not be)
    real_phase T7.1: store xi as a REAL N-vector + one phase scalar per slot,
               at `xi_dtype`'s component width (complex64 -> float32). Halves
               the xi term again and is exact iff `real_phase_residual` is 0.
               Default OFF -- taking it forfeits the imaginary channel.
    """

    def __init__(self, xi_dtype=np.complex64, p_floor=0.0, rank=None,
                 meta_dtype=np.float32, real_phase=False):
        xi_dtype, meta_dtype = np.dtype(xi_dtype), np.dtype(meta_dtype)
        if xi_dtype not in [np.dtype(d) for d in XI_DTYPES]:
            raise ValueError(f"xi_dtype {xi_dtype} not in {XI_DTYPES}")
        if meta_dtype not in [np.dtype(d) for d in META_DTYPES]:
            raise ValueError(f"meta_dtype {meta_dtype} not in {META_DTYPES}")
        if p_floor < 0:
            raise ValueError("p_floor must be >= 0")
        if rank is not None and rank < 1:
            raise ValueError("rank must be >= 1 or None")
        if real_phase and rank is not None:
            raise ValueError("real_phase and rank are alternative xi layouts")
        self.xi_dtype, self.p_floor = xi_dtype, float(p_floor)
        self.rank, self.meta_dtype = rank, meta_dtype
        self.real_phase = bool(real_phase)

    @property
    def component_dtype(self):
        """The real component width implied by `xi_dtype` (the real+phase
        layout's storage dtype)."""
        return np.dtype(np.float32 if self.xi_dtype == np.dtype(np.complex64)
                        else np.float64)

    @property
    def lossless(self):
        """True when this spec cannot change a single stored value."""
        return (self.xi_dtype == np.dtype(WORK_XI) and self.p_floor == 0.0
                and self.rank is None and not self.real_phase)

    def label(self):
        bits = ["c64" if self.xi_dtype == np.dtype(np.complex64) else "c128",
                "Psparse" if self.p_floor == 0 else f"Pfloor>{self.p_floor:g}"]
        if self.real_phase:
            bits.append("realphase")
        if self.rank is not None:
            bits.append(f"rank{self.rank}")
        if self.meta_dtype == np.dtype(np.float32):
            bits.append("meta32")
        return "+".join(bits)

    def __repr__(self):
        return f"CompressionSpec({self.label()})"


class CompressedState:
    """A narrowed encoding of one organism's memory store.

    Holds only what `store_bytes` accounts (xi, P, count, age) plus the
    shape/lifecycle bookkeeping needed to rebuild them (`used`, K, N).
    Everything else about the organism -- z, rng, evictions -- is the
    caller's business (E3 carries them at full width; they are O(N + K)).
    """

    def __init__(self, K, N, spec, xi=None, factors=None, used=None,
                 p_data=None, p_indices=None, p_indptr=None,
                 count=None, age=None, p_mass_kept=1.0, polar=None):
        self.K, self.N, self.spec = K, N, spec
        self.xi, self.factors, self.used = xi, factors, used
        self.polar = polar               # (real K x N, phase K) or None
        self.p_data, self.p_indices, self.p_indptr = p_data, p_indices, p_indptr
        self.count, self.age = count, age
        self.p_mass_kept = p_mass_kept

    # -- byte accounting (same four fields as store_bytes) ----------------
    @property
    def xi_bytes(self):
        if self.polar is not None:
            r, phi = self.polar
            return int(r.nbytes + phi.nbytes)
        if self.factors is not None:
            U, V = self.factors
            return int(U.nbytes + V.nbytes)
        return int(self.xi.nbytes)

    @property
    def p_bytes(self):
        return int(self.p_data.nbytes + self.p_indices.nbytes
                   + self.p_indptr.nbytes)

    @property
    def meta_bytes(self):
        return int(self.count.nbytes + self.age.nbytes)

    @property
    def nbytes(self):
        return self.xi_bytes + self.p_bytes + self.meta_bytes

    # -- reconstruction ---------------------------------------------------
    def xi_full(self):
        """xi at compute width (complex128), reconstructed."""
        if self.polar is not None:
            r, phi = self.polar
            return (r.astype(np.float64).astype(WORK_XI)
                    * np.exp(1j * phi.astype(np.float64))[:, None])
        if self.factors is not None:
            U, V = self.factors
            return (U.astype(WORK_XI) @ V.astype(WORK_XI))
        return self.xi.astype(WORK_XI)

    def P_full(self):
        """The dense K x K count matrix, reconstructed from CSR."""
        P = np.zeros((self.K, self.K), dtype=WORK_META)
        for i in range(self.K):
            lo, hi = int(self.p_indptr[i]), int(self.p_indptr[i + 1])
            if hi > lo:
                P[i, self.p_indices[lo:hi]] = self.p_data[lo:hi]
        return P

    def apply(self, org):
        """Write this store back into a live organism at compute width.

        The organism is left ready to perceive: xi/P/count/age are replaced,
        every other attribute (z, rng, evictions, graph wiring) untouched.
        """
        if (org.K, org.N) != (self.K, self.N):
            raise ValueError(f"shape mismatch: state is (K={self.K}, N={self.N}), "
                             f"organism is (K={org.K}, N={org.N})")
        org.xi = np.ascontiguousarray(self.xi_full())
        org.graph.P = np.ascontiguousarray(self.P_full())
        org.used = self.used.copy()
        org.count = self.count.astype(WORK_META)
        org.age = self.age.astype(WORK_META)
        return org


def compress(org, spec=None, **kw):
    """Compress an organism's memory store. `spec` or CompressionSpec kwargs."""
    if spec is None:
        spec = CompressionSpec(**kw)
    elif kw:
        raise TypeError("pass either a spec or CompressionSpec kwargs, not both")
    K, N = org.K, org.N
    xi = np.ascontiguousarray(org.xi, dtype=WORK_XI)

    # -- xi: dtype narrowing, low-rank factorization, or real+phase ------
    factors, xi_out, polar = None, None, None
    if spec.real_phase:
        # de-rotate each slot by the angle that minimizes its imaginary
        # energy (see real_phase_residual), store the real part + that angle
        comp = spec.component_dtype
        theta = 0.5 * np.angle((xi ** 2).sum(1))
        polar = ((xi * np.exp(-1j * theta)[:, None]).real.astype(comp),
                 theta.astype(comp))
    elif spec.rank is not None:
        # rank of a K x N bank is <= min(K, N); truncating at or above that
        # is exact up to float error, below it is a real approximation
        U, s, Vh = np.linalg.svd(xi, full_matrices=False)
        r = int(min(spec.rank, s.size))
        factors = ((U[:, :r] * s[:r]).astype(spec.xi_dtype),
                   Vh[:r].astype(spec.xi_dtype))
    else:
        xi_out = xi.astype(spec.xi_dtype)

    # -- P: CSR above the count floor (floor 0 == lossless) --------------
    P = np.asarray(org.P, dtype=WORK_META)
    keep = P > spec.p_floor
    total = float(P.sum())
    mass_kept = 1.0 if total == 0 else float(P[keep].sum()) / total
    counts = keep.sum(1)
    p_indptr = np.zeros(K + 1, dtype=np.int32)
    np.cumsum(counts, out=p_indptr[1:])
    rows, cols = np.nonzero(keep)
    p_indices = cols.astype(np.int32)
    p_dtype = np.float32 if _f32_safe(P[keep]) else np.float64
    p_data = P[rows, cols].astype(p_dtype)

    # -- count / age: float32 only where it is exact ---------------------
    meta_dtype = spec.meta_dtype
    if meta_dtype == np.dtype(np.float32) and not (
            _f32_safe(org.count) and _f32_safe(org.age)):
        meta_dtype = np.dtype(WORK_META)     # decline silently-lossy narrowing

    return CompressedState(
        K=K, N=N, spec=spec, xi=xi_out, factors=factors, polar=polar,
        used=np.asarray(org.used, dtype=bool).copy(),
        p_data=p_data, p_indices=p_indices, p_indptr=p_indptr,
        count=np.asarray(org.count).astype(meta_dtype),
        age=np.asarray(org.age).astype(meta_dtype),
        p_mass_kept=mass_kept)


def roundtrip(org, spec=None, **kw):
    """compress -> apply, in place: the persistence path a product takes
    between sessions, with the loss (if any) actually landed in the live
    organism. Returns the CompressedState so the caller can price it."""
    st = compress(org, spec=spec, **kw)
    st.apply(org)
    return st


def compression_report(org, spec):
    """(uncompressed_bytes, compressed_bytes, ratio, xi_rel_err) for a spec
    -- the measurement phase 33g's cost curve is built from. Does not touch
    the organism."""
    st = compress(org, spec=spec)
    raw = store_bytes(org)
    xi0 = np.asarray(org.xi, dtype=WORK_XI)
    denom = np.abs(xi0).max()
    err = float(np.abs(st.xi_full() - xi0).max() / (denom if denom else 1.0))
    return raw, st.nbytes, raw / st.nbytes, err


if __name__ == "__main__":
    # smoke: lossless spec is exact, c64 spec is tiny-error and ~2x smaller
    from organism import Organism

    rng = np.random.default_rng(0)
    N, K = 64, 40
    org = Organism(N=N, K=K, seed=0)
    org.used[:] = True
    org.count[:] = rng.integers(1, 80, K).astype(float)
    org.age[:] = rng.integers(0, 5000, K).astype(float)
    P = rng.integers(0, 4, (K, K)).astype(float)
    P[P < 3] = 0.0
    org.P = P

    for spec in (CompressionSpec(xi_dtype=np.complex128, meta_dtype=np.float64),
                 CompressionSpec(xi_dtype=np.complex128),
                 CompressionSpec(),
                 CompressionSpec(p_floor=2.0),
                 CompressionSpec(rank=16),
                 CompressionSpec(real_phase=True)):
        raw, comp, ratio, err = compression_report(org, spec)
        st = compress(org, spec=spec)
        exact = (np.array_equal(st.P_full(), org.P) and err == 0.0)
        print(f"  {spec.label():<28} {raw / 1e3:6.1f}KB -> {comp / 1e3:6.1f}KB "
              f"({ratio:4.2f}x)  xi relerr {err:.2e}  P mass {st.p_mass_kept:.4f}"
              f"  {'LOSSLESS' if exact else ''}")
