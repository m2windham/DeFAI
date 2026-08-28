"""Phase 54 -- DECISION EXPERIMENT 1 of 3: can a phase-aware readout read a
channel that `np.abs` cannot?

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

=============================================================================
WHY THIS ONE FIRST
=============================================================================
Phase 43 measured the readout's resolution floor at R ~ 0.06 and the store
at R = 2.55e-04, ~230x inside it, and concluded the imaginary channel is
empty. The owner's 2026-08-28 audit sharpened that: the invariance is an
exact GAUGE SYMMETRY of the whole mechanism, and the gauge fixing is the
de-rotation by angle(o[k]) before every write.

But nobody has ever changed the readout. Every consumer takes np.abs, which
is gauge-invariant by construction, so no measurement in this project has
ever been able to see the channel even in principle.

FHRR (Fourier Holographic Reduced Representations) reads it: similarity is
the MEAN COSINE OF COMPONENTWISE ANGLE DIFFERENCES, which is NOT
gauge-invariant. That is the point -- it is sensitive to exactly what abs
discards.

=============================================================================
THE DESIGN
=============================================================================
Sweep omega. At each value, train one organism and score the SAME bank two
ways:
  MAGNITUDE  |<xi_k, q>| / N          -- what every consumer does today
  FHRR       mean_i cos(arg(xi_k,i) - arg(q_i))  -- phase-sensitive
Task is next-symbol prediction on a stream with genuine ORDER structure
(the phase 51/52 process), so a readout that can see order has something to
win with.

R is reported at every omega from `real_phase_residual`, so the readout
result is anchored to the channel's measured occupancy rather than assumed.

=============================================================================
FIRST RUN VOID on P3, 2026-08-28 -- setup error, corrected before the
committed run. Recorded rather than quietly fixed.
=============================================================================
The first run reported median R = 4.4e-01 at EVERY omega including 0.15 --
flat, and three orders of magnitude above phase 43's measured 2.55e-04 at
the committed omega. P3 exists to catch exactly that and did.

The cause: the phase 51/52 codebook is COMPLEX-VALUED (standard_normal +
1j*standard_normal). Phase 43's 54-file census established that no committed
pipeline in this repo builds a complex stream -- every one casts real
embeddings -- so a complex-input stream fills the imaginary channel at the
input and the omega sweep then tests nothing about channel occupancy. The
kill rule fired on an invalid setup and that verdict is withdrawn.

CORRECTED: REAL-valued codebook cast to complex, which is what every
committed pipeline actually feeds. R then starts near zero and the sweep
measures what it was meant to measure.

SCOPE NOTE THAT NOW TRAVELS TO PHASES 51/52/56: those three used the complex
codebook too. Phase 51's headline is unaffected -- its APPEARANCE arm used
ground-truth appearance ids, so the input class does not enter the partition
comparison. Phases 52 and 56 measured slot INJECTIVITY on the complex stream,
and that caveat is now on the record for both.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  At the COMMITTED omega (0.15), FHRR does NOT beat MAGNITUDE. The channel
    is measured empty there; a gain would mean the metric is picking up
    something other than stored phase and the phase is void.
P2  At omega >= 4, where phase 43 measured R = 0.33 (above the 0.06 floor),
    FHRR BEATS MAGNITUDE by >= 0.02 accuracy, sign-consistent on all five
    held-out seeds. **This is the decisive clause.** If it fails, the phase
    channel is unreadable even when FULL, G2 does not close, and the
    complex representation cannot be justified on capability grounds either.
P3  R rises with omega as omega^2 and crosses 0.06 between omega=1 and 4,
    reproducing phase 43 on this protocol.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT
=============================================================================
Raising omega destroys the ATTRACTOR structure, so any FHRR gain at high
omega is bought by wrecking the magnitude readout rather than by reading
phase. CONTROL: report MAGNITUDE's own accuracy across omega. If magnitude
collapses while FHRR merely declines less, M1 has fired and the result is
NOT a demonstration that phase carries usable information.

=============================================================================
KILL RULE
=============================================================================
If P2 fails, this arm is finished: the imaginary channel is not worth the
2.03x it costs on capability grounds, and no future target may propose a
phase-carrying mechanism without first clearing this bar.
"""
import numpy as np
from phase51_causal_state_recruit import (N_DIM, hidden_walk, emit,
                                          STATE_APPEARANCE, N_APPEARANCE)
from organism_compress import real_phase_residual

OMEGAS = (0.15, 0.5, 1.0, 4.0, 8.0)
K = 24
N_TRAIN, N_TEST = 4000, 4000


def real_codebook(rng):
    """REAL directions cast to complex -- what every committed pipeline
    feeds (phase 43's 54-file census). A complex codebook fills the
    imaginary channel at the input and voids the omega sweep."""
    M = rng.standard_normal((N_APPEARANCE, N_DIM))
    Q, _ = np.linalg.qr(M.T)
    return (Q.T[:N_APPEARANCE] * np.sqrt(N_DIM)).astype(complex)


def _fhrr_sim(xi_live, q):
    """FHRR similarity: mean over components of cos(angle difference).
    NOT gauge-invariant -- reads exactly what np.abs throws away."""
    a = np.angle(xi_live)[None, :, :]          # (1, S, N)
    b = np.angle(q)[:, None, :]                # (Q, 1, N)
    return np.cos(a - b).mean(axis=2)          # (Q, S)


def _mag_sim(xi_live, q):
    return np.abs(q @ xi_live.conj().T) / xi_live.shape[1]


def run_seed(seed, omega):
    from organism import Organism
    rng = np.random.default_rng(4242 + seed)
    code = real_codebook(np.random.default_rng(4242 + 900 + seed))
    tr = hidden_walk(N_TRAIN, rng); te = hidden_walk(N_TEST, rng)
    tr_f, tr_a = emit(tr, code, rng); te_f, te_a = emit(te, code, rng)

    org = Organism(N=N_DIM, K=K, seed=seed, omega=omega, backend="numpy")
    org.perceive(tr_f)
    live = np.flatnonzero(org.used)
    if len(live) < 2:
        return None
    xi = org.xi[live]
    R = float(np.median(real_phase_residual(xi)))

    # queries are the settled field is not available post-hoc, so use the
    # emitted frame block's last frame -- the same thing the bank was built
    # from, which is the train/query match the audit found missing elsewhere
    def blocks(frames, app):
        H = len(frames) // len(app)
        return np.array([frames[i * H + H - 1] for i in range(len(app))])

    qtr, qte = blocks(tr_f, tr_a), blocks(te_f, te_a)
    out = {}
    for name, sim in (("MAGNITUDE", _mag_sim), ("FHRR", _fhrr_sim)):
        # label each slot by the next symbol most often following it (train),
        # then score next-symbol prediction on held-out
        s_tr = np.argmax(sim(xi, qtr), axis=1)
        tab = np.zeros((len(live), N_APPEARANCE))
        for i in range(len(tr_a) - 1):
            tab[s_tr[i], tr_a[i + 1]] += 1
        lab = tab.argmax(1)
        s_te = np.argmax(sim(xi, qte), axis=1)
        acc = float((lab[s_te[:-1]] == te_a[1:]).mean())
        out[name] = acc
    out["R"] = R
    return out


def main():
    print("=" * 78)
    print("PHASE 54 -- can a phase-aware readout read what np.abs cannot?")
    print("DECISION EXPERIMENT 1 of 3")
    print("=" * 78)
    sel, hout = range(0, 5), range(5, 10)
    res = {}
    for om in OMEGAS:
        res[om] = {s: run_seed(s, om) for s in list(sel) + list(hout)}

    print(f"\n{'omega':>7}{'median R':>12}{'MAGNITUDE':>12}{'FHRR':>10}"
          f"{'FHRR-MAG':>11}{'sign':>7}   held-out means")
    verdict = {}
    for om in OMEGAS:
        rows = [res[om][s] for s in hout if res[om][s]]
        if not rows:
            print(f"{om:>7}   (no live bank)"); continue
        R = np.mean([r["R"] for r in rows])
        m = np.array([r["MAGNITUDE"] for r in rows])
        f = np.array([r["FHRR"] for r in rows])
        d = f - m
        flag = "  <-- above readout floor R=0.06" if R > 0.06 else ""
        print(f"{om:>7}{R:>12.2e}{m.mean():>12.4f}{f.mean():>10.4f}"
              f"{d.mean():>+11.4f}{int((d>0).sum()):>5}/5{flag}")
        verdict[om] = (d, m, R)

    print("\n--- P1: at the COMMITTED omega=0.15, FHRR must NOT beat MAGNITUDE " + "-" * 6)
    d, m, R = verdict[0.15]
    p1 = not ((d > 0).all() and d.mean() >= 0.02)
    print(f"  delta {d.mean():+.4f}, positive {int((d>0).sum())}/5, R={R:.2e}")
    print(f"  P1: {'HELD (channel is empty there, as measured)' if p1 else 'FAILED'}")

    print("\n--- P2 (DECISIVE): at omega>=4, FHRR must beat MAGNITUDE by >=0.02 " + "-" * 4)
    p2 = False
    for om in (4.0, 8.0):
        if om not in verdict:
            continue
        d, m, R = verdict[om]
        ok = (d > 0).all() and d.mean() >= 0.02
        p2 = p2 or ok
        print(f"  omega={om}: delta {d.mean():+.4f}, positive {int((d>0).sum())}/5,"
              f" R={R:.2e}  -> {'CLEARS' if ok else 'does not clear'}")
    print(f"  P2: {'HELD' if p2 else 'FAILED'}")

    print("\n--- M1: is any high-omega gain bought by wrecking MAGNITUDE? " + "-" * 9)
    m0 = verdict[0.15][1].mean()
    for om in (4.0, 8.0):
        if om in verdict:
            print(f"  MAGNITUDE at omega={om}: {verdict[om][1].mean():.4f}"
                  f"  (vs {m0:.4f} at 0.15)")
    print("  If MAGNITUDE collapses and FHRR merely declines less, M1 has FIRED")
    print("  and this is NOT evidence that phase carries usable information.")

    print("\n--- KILL RULE " + "-" * 58)
    if not p2:
        print("  P2 FAILED -> the phase channel is UNREADABLE EVEN WHEN FULL.")
        print("  G2 does not close. The complex representation cannot be")
        print("  justified on capability grounds. No future target may propose")
        print("  a phase-carrying mechanism without first clearing this bar.")
    else:
        print("  P2 held: a phase-aware readout reads what np.abs cannot.")
    print("=" * 78)


if __name__ == "__main__":
    main()
