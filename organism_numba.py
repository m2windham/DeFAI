"""
E2: NUMBA BACKEND -- compatibility seam over the unified fastpath.

History: two sessions built E2 in parallel on 2026-07-13/15. This module's
original op-for-op kernels (landed first, 8x on pool+amb perceive at
N=30/K=60) were superseded by `fastpath.py` (BLAS overlap matvec with a
maintained conjugate, squared-magnitude argmax, bounded-memory chunked
streaming so generator streams never materialize -- 13.0x measured at the
true phase-23 corpus shape, K_CAP=1580 / 4.9M frames). The two were unified
rather than kept side by side; the ROADMAP E2 row records both.

The public seam is unchanged and still pinned by the harness:
  - `NumbaOrganism` remains a drop-in Organism whose hot loops run JIT'd --
    it is now a thin alias forcing `backend="numba"` on the unified
    dispatch in organism.py (which any plain Organism also uses when
    DEFAI_BACKEND=numba, or by default under "auto" when numba is
    installed);
  - `DEFAI_BACKEND=numba regression_harness.py` still selects it;
  - E3's cross-backend restore check (harness section 8) still constructs
    this class via organism_state.load_state(cls=NumbaOrganism).
"""

from organism import Organism


class NumbaOrganism(Organism):
    """Drop-in Organism with JIT hot loops (unified E2 fastpath). Same API,
    same state layout, same TransitionGraph/EventBoundary seam. Construction
    fails if numba is not installed -- this class is the explicit request
    for the JIT backend, so silent numpy fallback would be a lie."""

    def __init__(self, N=128, K=8, omega=0.25, beta=12.0, seed=0, backend=None):
        super().__init__(N=N, K=K, omega=omega, beta=beta, seed=seed,
                         backend="numba")
