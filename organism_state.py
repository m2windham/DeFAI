"""
E3: STATE SERIALIZATION -- save/load of the full organism, with a schema
version, so stored memories survive mechanism upgrades. Not product polish:
an episodic-memory product is DEFINED by persistence (ROADMAP, engineering
track). This is the last fork-gate item.

Guarantees (each one pinned in regression_harness.py section 8):
  - LOSSLESS: save -> load -> continue perceiving is BITWISE identical to
    never having stopped (same backend). The field state z, the memory bank
    xi, the logic layer's transition graph P, usage counts, and the rng
    stream all round-trip exactly.
  - DETERMINISTIC REPLAY: the rng generator state is captured, so a restored
    organism's recall() produces the identical stochastic trajectory the
    original would have.
  - SCHEMA-VERSIONED: files carry SCHEMA_VERSION; loading a file written by
    a different schema raises with both versions named, instead of silently
    misreading state across mechanism upgrades.
  - BACKEND-AGNOSTIC: state saved from the reference Organism loads into
    NumbaOrganism and vice versa (identical state layout by construction --
    E2's backend contract).

Bounded memory: the slot cap is K by construction, and eviction is the
mechanism's own use-it-or-lose-it recycling (phase 17), extended by T1.2's
slot budget (eviction under recruitment pressure, `evict > 0`). The budget's
staleness clock (org.age) and per-slot eviction tally (org.evictions) are
part of the organism's dynamical state and round-trip here. Files written
before T1.2 lack the two arrays and load with zeroed clocks -- exactly the
state a pre-budget organism had -- so the schema version is unchanged
(additive migration, not a reinterpretation of existing fields).

Consolidation products (mem/Pn/kept_idx) are saved when present so a
restored organism can recall() immediately without re-consolidating.

SCHEMA v2 (T1.8, phase 33g): compressed stores
----------------------------------------------
v2 adds an OPTIONAL compressed encoding of the memory store (narrowed `xi`
dtype, CSR transition matrix, narrowed count/age -- see
`organism_compress.py`), written by `save_state(org, path, compress=spec)`.

  - v2 WITHOUT a spec is byte-for-byte the v1 layout with the version
    number bumped: the lossless round-trip pins are unchanged and still
    bitwise. Every array v1 wrote, v2 writes.
  - v1 FILES STILL LOAD. `load_state` accepts any version in
    SUPPORTED_SCHEMAS and reads v1's field set directly (v1 has no
    compression marker, so the uncompressed path is taken).
  - a COMPRESSED file is lossy exactly where its spec says it is (complex64
    xi, a count floor > 0, low-rank factors) and lossless otherwise; it
    always restores at compute width, so a loaded organism perceives at
    complex128 regardless of how narrowly it was stored.

The lossy case is deliberately NOT allowed to masquerade as the bitwise
guarantee above: `save_state` records the spec in the file, and harness
section 8 pins the uncompressed round-trip at zero delta and the compressed
one inside a stated tolerance.

SCHEMA v3 (T3.3): the symbol registry
-------------------------------------
v3 adds the OPTIONAL `SymbolRegistry` (organism.py) -- stable symbol IDs
decoupled from slot indices. Persistence is the whole point of an identity
layer: an ID handed to a caller before a save must still name the same
memory after the load, or the registry is only as stable as one process.

  - v3 WITHOUT a registry is byte-for-byte the v2 layout with the version
    bumped. Every array v2 wrote, v3 writes; the bitwise round-trip pins are
    unchanged. The compressed encoding is likewise untouched -- the registry
    is O(K) plus the alias/tombstone tables and is never a byte problem.
  - v1 AND v2 FILES STILL LOAD, into an organism with no registry, which is
    exactly the state a pre-T3.3 organism had (additive migration -- no
    existing field is reinterpreted). `save_state_v1` / `save_state_v2`
    write the older layouts so the backward-load guarantee is tested
    against real files rather than an assertion about them.
  - the registry round-trip is BITWISE and complete: slot bindings, the ID
    counter, the alias chains, the tombstones, and the consolidation view.
    A restored organism keeps minting from where it stopped, so an ID is
    never reissued across a save boundary.
"""

import json
import numpy as np

from organism import Organism
from organism_compress import (
    CompressedState, CompressionSpec, WORK_XI, compress,
)

SCHEMA_VERSION = 3
SUPPORTED_SCHEMAS = (1, 2, 3)


def _pack_registry(org):
    """T3.3 registry -> flat arrays (no pickle). Empty dict when the organism
    has no registry, so v3 files without one are byte-identical to v2."""
    reg = getattr(org, 'registry', None)
    if reg is None:
        return {}
    alias = sorted(reg.alias.items())
    out = dict(
        reg_slot_sym=np.asarray(reg.slot_sym, dtype=np.int64),
        reg_next_id=np.asarray(reg.next_id, dtype=np.int64),
        reg_alias=np.array(alias, dtype=np.int64).reshape(-1, 2),
        reg_dead=np.array(sorted(reg.dead), dtype=np.int64),
    )
    if reg.mem_index:                    # the consolidation view, when present
        items = sorted(reg.mem_index.items())
        out['reg_mem_index'] = np.array(items, dtype=np.int64).reshape(-1, 2)
    return out


def _unpack_registry(f, org):
    """Rebuild the registry from a v3 file. Absent (v1/v2, or a v3 written
    without one) leaves org.registry as the constructor made it: None."""
    if 'reg_slot_sym' not in f:
        return
    reg = org.enable_symbols()
    reg.slot_sym = f['reg_slot_sym'].copy()
    reg.next_id = f['reg_next_id'].copy()
    reg.alias = {int(a): int(b) for a, b in f['reg_alias']}
    reg.dead = {int(d) for d in f['reg_dead']}
    reg.mem_index = ({int(s): int(r) for s, r in f['reg_mem_index']}
                     if 'reg_mem_index' in f else {})


def save_state(org, path, compress_spec=None):
    """Serialize the full organism state to `path` (.npz, no pickle).

    compress_spec: None (default) writes the full-width v1-compatible
    layout. A `CompressionSpec` writes the narrowed store instead -- lossy
    exactly where the spec is lossy, and always restored at compute width.
    """
    if compress_spec is not None:
        return _save_compressed(org, path, compress_spec)
    arrays = dict(
        schema=np.array([SCHEMA_VERSION]),
        params=np.array([org.N, org.K], dtype=np.int64),
        hyper=np.array([org.omega, org.beta], dtype=np.float64),
        xi=org.xi, used=org.used, count=org.count,
        age=org.age, evictions=org.evictions,
        P=org.graph.P, z=org.z,
        rng_state=np.frombuffer(
            json.dumps(org.rng.bit_generator.state).encode(), dtype=np.uint8),
    )
    arrays.update(_pack_registry(org))
    if hasattr(org, 'mem'):
        arrays['mem'] = org.mem
        arrays['Pn'] = org.Pn
        arrays['kept_idx'] = np.array(org.kept_idx, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def save_state_v2(org, path):
    """Write the PRE-T3.3 (schema v2) layout -- same role as `save_state_v1`:
    a real older file to test the backward-load guarantee against. The
    registry is dropped, which is the honest v2 semantics (v2 had none)."""
    arrays = dict(
        schema=np.array([2]),
        params=np.array([org.N, org.K], dtype=np.int64),
        hyper=np.array([org.omega, org.beta], dtype=np.float64),
        xi=org.xi, used=org.used, count=org.count,
        age=org.age, evictions=org.evictions,
        P=org.graph.P, z=org.z,
        rng_state=np.frombuffer(
            json.dumps(org.rng.bit_generator.state).encode(), dtype=np.uint8),
    )
    if hasattr(org, 'mem'):
        arrays['mem'] = org.mem
        arrays['Pn'] = org.Pn
        arrays['kept_idx'] = np.array(org.kept_idx, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def save_state_v1(org, path):
    """Write the PRE-T1.8 (schema v1) layout. Kept so the backward-load
    guarantee is testable against a real v1 file rather than an assertion
    about one -- exercised by harness section 8 and
    `test_fastpath_equivalence.py`. Not for production saves; use
    `save_state`."""
    arrays = dict(
        schema=np.array([1]),
        params=np.array([org.N, org.K], dtype=np.int64),
        hyper=np.array([org.omega, org.beta], dtype=np.float64),
        xi=org.xi, used=org.used, count=org.count,
        age=org.age, evictions=org.evictions,
        P=org.graph.P, z=org.z,
        rng_state=np.frombuffer(
            json.dumps(org.rng.bit_generator.state).encode(), dtype=np.uint8),
    )
    np.savez_compressed(path, **arrays)


def _save_compressed(org, path, spec):
    """v2 compressed layout: the memory store (xi, P, count, age) in its
    narrowed encoding, everything else (z, rng, evictions, consolidation
    products) at full width -- those are O(N + K) and not the byte problem
    T1.8 measured."""
    st = compress(org, spec=spec)
    arrays = dict(
        schema=np.array([SCHEMA_VERSION]),
        params=np.array([org.N, org.K], dtype=np.int64),
        hyper=np.array([org.omega, org.beta], dtype=np.float64),
        # compression marker + spec, so load reconstructs without guessing
        compressed=np.array([1], dtype=np.int64),
        c_spec=np.frombuffer(json.dumps(dict(
            xi_dtype=str(np.dtype(spec.xi_dtype)), p_floor=float(spec.p_floor),
            rank=(None if spec.rank is None else int(spec.rank)),
            meta_dtype=str(np.dtype(spec.meta_dtype)),
            lowrank=st.factors is not None,
        )).encode(), dtype=np.uint8),
        p_data=st.p_data, p_indices=st.p_indices, p_indptr=st.p_indptr,
        used=st.used, count=st.count, age=st.age,
        evictions=org.evictions, z=org.z,
        rng_state=np.frombuffer(
            json.dumps(org.rng.bit_generator.state).encode(), dtype=np.uint8),
    )
    arrays.update(_pack_registry(org))    # O(K): never a byte problem
    if st.factors is not None:
        arrays['xi_U'], arrays['xi_V'] = st.factors
    else:
        arrays['xi'] = st.xi
    if hasattr(org, 'mem'):
        arrays['mem'] = np.asarray(org.mem).astype(spec.xi_dtype)
        arrays['Pn'] = org.Pn
        arrays['kept_idx'] = np.array(org.kept_idx, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def _load_compressed(f, org, K, N):
    """Rebuild the store from a v2 compressed file, at COMPUTE width."""
    spec_d = json.loads(bytes(f['c_spec']).decode())
    spec = CompressionSpec(xi_dtype=np.dtype(spec_d['xi_dtype']),
                           p_floor=spec_d['p_floor'], rank=spec_d['rank'],
                           meta_dtype=np.dtype(spec_d['meta_dtype']))
    factors = ((f['xi_U'].copy(), f['xi_V'].copy())
               if spec_d['lowrank'] else None)
    st = CompressedState(
        K=K, N=N, spec=spec,
        xi=(None if factors is not None else f['xi'].copy()), factors=factors,
        used=f['used'].copy(), p_data=f['p_data'].copy(),
        p_indices=f['p_indices'].copy(), p_indptr=f['p_indptr'].copy(),
        count=f['count'].copy(), age=f['age'].copy())
    st.apply(org)
    return org


def load_state(path, cls=Organism):
    """Restore an organism (of `cls` -- Organism or NumbaOrganism) from
    `path`. Raises ValueError on an unsupported schema version.

    Backward compatible: v1 files (written before T1.8) and v2 files
    (written before T3.3) load unchanged -- the older file simply carries no
    registry, and the restored organism has none, which is what it had.
    Compressed files restore at compute width (complex128/float64), so the
    organism they produce is immediately perceivable on either backend.
    """
    with np.load(path, allow_pickle=False) as f:
        ver = int(f['schema'][0])
        if ver not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f"state file schema v{ver} not in supported {SUPPORTED_SCHEMAS}; "
                "write a migration, don't guess")
        N, K = (int(v) for v in f['params'])
        omega, beta = (float(v) for v in f['hyper'])
        org = cls(N=N, K=K, omega=omega, beta=beta, seed=0)
        if 'compressed' in f and int(f['compressed'][0]):
            _load_compressed(f, org, K, N)
        else:
            org.xi = f['xi'].copy()
            org.used = f['used'].copy()
            org.count = f['count'].copy()
            # pre-T1.2 files carry no budget state; a zeroed clock is exactly
            # the state a pre-budget organism had (additive migration)
            org.age = f['age'].copy() if 'age' in f else np.zeros(K)
            org.graph.P = f['P'].copy()
        org.evictions = f['evictions'].copy() if 'evictions' in f else np.zeros(K)
        org.z = f['z'].copy()
        org.rng.bit_generator.state = json.loads(bytes(f['rng_state']).decode())
        _unpack_registry(f, org)          # v3 only; absent in v1/v2
        if 'mem' in f:
            org.mem = f['mem'].astype(WORK_XI)
            org.Pn = f['Pn'].copy()
            org.kept_idx = [int(i) for i in f['kept_idx']]
    return org


if __name__ == "__main__":
    # smoke: stop/restore mid-stream must equal never stopping
    import tempfile, os
    rng = np.random.default_rng(3)
    N, H, K = 64, 3, 6
    NORM = np.sqrt(N)
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    T = np.array([[0.0, 0.8, 0.2], [0.2, 0.0, 0.8], [0.8, 0.2, 0.0]])

    def stream(n, seed):
        r = np.random.default_rng(seed)
        h = 0; out = []
        for i in range(n):
            if i % 40 == 0 and i > 0:
                h = r.choice(H, p=T[h])
            out.append(G[h] + 0.4 * (r.standard_normal(N) + 1j * r.standard_normal(N)))
        return out

    s1, s2 = stream(8000, 1), stream(8000, 2)

    a = Organism(N=N, K=K, seed=0)
    a.perceive(s1); a.perceive(s2); a.consolidate()

    b = Organism(N=N, K=K, seed=0)
    b.perceive(s1)
    path = os.path.join(tempfile.gettempdir(), "e3_smoke.npz")
    save_state(b, path)
    c = load_state(path)
    c.perceive(s2); c.consolidate()

    print("continue-after-restore vs never-stopped:")
    print(f"  max |dxi| = {np.abs(a.xi - c.xi).max():.2e}")
    print(f"  max |dP|  = {np.abs(a.P - c.P).max():.2e}")
    ra, rc = a.recall(5000), c.recall(5000)
    same = len(ra) == len(rc) and bool(np.all(ra == rc))
    print(f"  recall sequences identical: {same} ({len(ra)} hops)")
    print("verdict:", "LOSSLESS + deterministic replay"
          if np.abs(a.xi - c.xi).max() == 0 and same else "FAIL -- state leaks")

    # -- v2 additions (T1.8): backward load + compressed stores ----------
    print("\nschema v2 (T1.8):")
    v1p = os.path.join(tempfile.gettempdir(), "e3_v1.npz")
    save_state_v1(b, v1p)
    d = load_state(v1p)
    print(f"  v1 file loads under v{SCHEMA_VERSION}: "
          f"max |dxi| = {np.abs(d.xi - b.xi).max():.2e}, "
          f"max |dP| = {np.abs(d.P - b.P).max():.2e}, "
          f"age/evictions restored = {np.array_equal(d.age, b.age)}")

    for spec in (CompressionSpec(xi_dtype=np.complex128, meta_dtype=np.float64),
                 CompressionSpec()):
        cp = os.path.join(tempfile.gettempdir(), "e3_c.npz")
        save_state(b, cp, compress_spec=spec)
        e = load_state(cp)
        dxi = np.abs(e.xi - b.xi).max()
        dP = np.abs(e.P - b.P).max()
        print(f"  {spec.label():<28} bytes {os.path.getsize(cp):>7}  "
              f"max |dxi| {dxi:.2e}  max |dP| {dP:.2e}  "
              f"restores at {e.xi.dtype}"
              f"{'  (LOSSLESS)' if dxi == 0 and dP == 0 else ''}")
    print(f"  uncompressed file bytes: {os.path.getsize(path)}")

    # -- v3 additions (T3.3): the symbol registry round-trips -------------
    print("\nschema v3 (T3.3): symbol registry")
    g = Organism(N=N, K=K, seed=0, symbols=True)
    g.perceive(s1); g.perceive(s2); g.consolidate()
    gp = os.path.join(tempfile.gettempdir(), "e3_v3.npz")
    save_state(g, gp)
    h = load_state(gp)
    reg, back = g.registry, h.registry
    exact = (np.array_equal(reg.slot_sym, back.slot_sym)
             and int(reg.next_id[0]) == int(back.next_id[0])
             and reg.alias == back.alias and reg.dead == back.dead
             and reg.mem_index == back.mem_index)
    print(f"  {reg.minted()} IDs minted, {len(reg.live())} live, "
          f"{len(reg.alias)} fused, {len(reg.dead)} tombstoned")
    print(f"  registry round-trip exact: {exact}")
    v2p = os.path.join(tempfile.gettempdir(), "e3_v2.npz")
    save_state_v2(g, v2p)
    i = load_state(v2p)
    print(f"  v2 file loads under v{SCHEMA_VERSION}: "
          f"max |dxi| = {np.abs(i.xi - g.xi).max():.2e}, "
          f"registry absent (as v2 had none) = {i.registry is None}")
    off = Organism(N=N, K=K, seed=0)
    off.perceive(s1); off.perceive(s2)
    print(f"  registry on-vs-off state drift: "
          f"{max(np.abs(g.xi - off.xi).max(), np.abs(g.P - off.P).max()):.2e}"
          "  (observational: must be 0)")
