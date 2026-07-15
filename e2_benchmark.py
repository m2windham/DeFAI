"""
E2 BENCHMARK -- perceive throughput at phase-23 scale, numpy vs numba.

Reproduces phase 23's training shape without the corpus dependency (the
Gutenberg books are ephemeral in /tmp): V=395 words, DIM=50 embeddings,
hold=4 frames per token, K_CAP=1580 slots, ~408K tokens per epoch, 3 epochs
(the 'recipe' arm, recruit=0.75 -- the configuration whose 3-epoch run cost
~15 minutes and motivated E2). The stream is a generator, as in phase 23;
frames never materialize.

The numpy backend is timed on a 100K-frame slice and extrapolated (running
it for the full 4.9M frames is exactly the cost this port removes); the
numba backend runs the full 3-epoch load for a real end-to-end number.

Run: python e2_benchmark.py
"""

import time

import numpy as np

import fastpath
from organism import normalize
from polysemy_organism import PolysemyOrganism

V, DIM, HOLD, N_TOKENS, EPOCHS = 395, 50, 4, 408000, 3
K_CAP = min(2000, V * 4)
NORM = np.sqrt(DIM)

rng = np.random.default_rng(0)
emb = rng.standard_normal((V, DIM))
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
emb_c = np.array([normalize(e.astype(complex), NORM) for e in emb])
# zipf-ish token sequence, like real text
p = 1.0 / np.arange(1, V + 1); p /= p.sum()
train_seq = rng.choice(V, size=N_TOKENS, p=p)


def make_stream(seq, n_tokens=None):
    for w in (seq if n_tokens is None else seq[:n_tokens]):
        for _ in range(HOLD):
            yield emb_c[w]


def run(backend, n_tokens, epochs):
    o = PolysemyOrganism(N=DIM, K=K_CAP, omega=0.15, beta=10.0, seed=0,
                         backend=backend)
    t0 = time.time()
    for _ in range(epochs):
        o.perceive(make_stream(train_seq, n_tokens), g_in=5.0, dt=0.05,
                   eta=0.02, recruit=0.75)
    dt = time.time() - t0
    frames = n_tokens * HOLD * epochs
    return o, dt, frames


print(f"E2 benchmark: perceive at phase-23 scale "
      f"(V={V}, DIM={DIM}, K_CAP={K_CAP}, {N_TOKENS} tokens x hold {HOLD} x {EPOCHS} epochs "
      f"= {N_TOKENS*HOLD*EPOCHS/1e6:.1f}M frames)\n")

if fastpath.HAVE_NUMBA:
    fastpath.warmup()

o_np, t_np, f_np = run("numpy", 25000, 1)
rate_np = f_np / t_np
full = N_TOKENS * HOLD * EPOCHS
print(f"numpy : {f_np/1e3:.0f}K frames in {t_np:.1f}s = {rate_np/1e3:.1f}K frames/s "
      f"-> full load projected {full/rate_np/60:.1f} min")

if fastpath.HAVE_NUMBA:
    o_nb, t_nb, f_nb = run("numba", N_TOKENS, EPOCHS)
    rate_nb = f_nb / t_nb
    print(f"numba : {f_nb/1e6:.2f}M frames in {t_nb:.1f}s = {rate_nb/1e3:.1f}K frames/s "
          f"(measured, full load)")
    print(f"\nspeedup: {rate_nb/rate_np:.1f}x  |  full 3-epoch perceive: "
          f"{full/rate_np/60:.1f} min (numpy, projected) -> {t_nb/60:.2f} min (numba, measured)")
    used = int(o_nb.used.sum())
    print(f"sanity: numba run recruited {used} slots (nonzero, bounded by K_CAP={K_CAP})")
else:
    print("numba not installed -- fastpath benchmark skipped")
