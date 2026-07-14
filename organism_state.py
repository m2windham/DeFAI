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
mechanism's own use-it-or-lose-it recycling (phase 17) -- there is no
separate eviction policy to serialize.

Consolidation products (mem/Pn/kept_idx) are saved when present so a
restored organism can recall() immediately without re-consolidating.
"""

import json
import numpy as np

from organism import Organism

SCHEMA_VERSION = 1


def save_state(org, path):
    """Serialize the full organism state to `path` (.npz, no pickle)."""
    arrays = dict(
        schema=np.array([SCHEMA_VERSION]),
        params=np.array([org.N, org.K], dtype=np.int64),
        hyper=np.array([org.omega, org.beta], dtype=np.float64),
        xi=org.xi, used=org.used, count=org.count,
        P=org.graph.P, z=org.z,
        rng_state=np.frombuffer(
            json.dumps(org.rng.bit_generator.state).encode(), dtype=np.uint8),
    )
    if hasattr(org, 'mem'):
        arrays['mem'] = org.mem
        arrays['Pn'] = org.Pn
        arrays['kept_idx'] = np.array(org.kept_idx, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_state(path, cls=Organism):
    """Restore an organism (of `cls` -- Organism or NumbaOrganism) from
    `path`. Raises ValueError on schema mismatch."""
    with np.load(path, allow_pickle=False) as f:
        ver = int(f['schema'][0])
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"state file schema v{ver} != supported v{SCHEMA_VERSION}; "
                "write a migration, don't guess")
        N, K = (int(v) for v in f['params'])
        omega, beta = (float(v) for v in f['hyper'])
        org = cls(N=N, K=K, omega=omega, beta=beta, seed=0)
        org.xi = f['xi'].copy()
        org.used = f['used'].copy()
        org.count = f['count'].copy()
        org.graph.P = f['P'].copy()
        org.z = f['z'].copy()
        org.rng.bit_generator.state = json.loads(bytes(f['rng_state']).decode())
        if 'mem' in f:
            org.mem = f['mem'].copy()
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
