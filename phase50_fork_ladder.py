"""
PHASE 50 (T7.7) -- FORK DECISION MEASUREMENT: THE (A) STORE ON THE LADDER.

WHAT THIS ASKS. T7.1 (phase 43) priced the storage fork and correctly
refused to choose it: (A) SPEND the imaginary channel -- store each slot as
a real N-vector plus one phase scalar (`CompressionSpec(real_phase=True)`,
default OFF) -- at 44.83 KB/ACC-pt = 1.318x the bar held-out (39.04 =
1.147x with T7.2's narrow CSR); or (B) KEEP the width at 76.10 = 2.236x
(70.30 = 2.066x narrow). The owner's ruling (2026-08-11) is to measure
before choosing: nobody has run the (A) store and the accuracy arm IN THE
SAME EXPERIMENT. Phase 43's <=0.005 behavior bound was measured on the
cost-frontier protocol's final mean ACC only; FORG -- exactly where a lossy
store would be expected to hurt first -- was never reported for the (A)
layout, and the ladder context (per-seed prototype bar, replay) was never
run alongside it. The 2026-08-11 re-scope removed the fork's gate stakes,
so this is an architecture decision on measured merit, and this phase must
end in a RECOMMENDATION (T7.1 was right to withhold one; by the end of
this run the axis it lacked is measured).

PROTOCOL. The 33c ladder protocol at K=112 (33h's floor arm) and K=160
(33d's bar-crossing arm): class-incremental split-digits, 5 tasks x 2
classes, single pass per task (epochs=3 within the task, hold=8, 33c's
verbatim recipe), evict=250, LabelEvidenceReadout with eviction
invalidation -- executed through `phase33h_cost_frontier.run_arm`
VERBATIM in STORE MODE (the store is compressed at every task boundary and
carried forward: the deployment persistence path, where loss compounds).
Arms per (K, seed):

  c64  spec = c64 + CSR(floor 0) + meta32          (33g's headline; branch B)
  rp   spec = c64 + CSR(floor 0) + meta32 + real_phase   (the (A) layout)

Both decoders are scored off the SAME trained organism per arm (33e's
technique): argmax (the ladder's committed decoder) and calib-b8 (33h's
floor decoder). The decoder choice cancels in the paired fork delta
(rp - c64 at fixed decoder); both are reported. Paired seeds s=0-4,
HELD-OUT confirmation s=5-9, and the prototype bar RECOMPUTED ON EACH
SEED'S OWN SPLIT (`run_prototypes_seed` -- an unpaired fixed-seed bar is
the exact bug T1.6/T1.8 were burned by). Replay is reseeded per split
(torch.manual_seed(seed) + build_split(seed), 33c's replay recipe
otherwise verbatim), which upgrades 33h's "direction, not a claim"
single-seed replay caveat into a per-seed measured arm. Paired deltas
carry percentile bootstrap CIs (B=20000 resamples over seeds). Every byte
number is computed as `live_bytes` under all four specs (c64/rp x
CSR/narrow-CSR) on each arm's own final store and carries its observation
count (phase 44's standing rule: density is not scale-free): frames fed,
sum(P) transitions recorded, nnz, density.

DISCLOSURE, per the honesty rules. Before this pre-registration was
committed, the work order's mandatory compute-budget probe ("time ONE seed
x one arm x one K first") ran seed 0 only: K=112 both arms, K=160 c64
only, bar seed 0, replay seed 0. Numbers seen by the author before
registration: K=112 c64 argmax 0.8784 / calib 0.9138 (FORG 0.0266), rp
argmax 0.8835 / calib 0.9081 (FORG 0.0237); K=160 c64 argmax 0.9002 /
calib 0.9301 (FORG 0.0124); bar 0.8721/120 protos; replay 0.9100/0.1058.
The predictions below were therefore written after seeing those cells and
BEFORE every other cell -- in particular before ANY rp run at K=160 and
before every held-out seed (s=5-9), which carry all confirmations.
Recorded so the registration is judged on what it could not have known.

======================================================================
PRE-REGISTRATION (committed before the committed run)
======================================================================

P1 -- REPRODUCTION ANCHORS. Exact (same code path, same seeds): bar seed 0
= 0.872 / 120 protos / 30720 B; K=160 c64 argmax seed 0 = 0.900 (33d/33g
committed; probe 0.9002); K=112 c64+CSR calib-b8 held-out floor = 76.10
KB/ACC-pt = 2.236x the bar and narrow CSR 70.30 = 2.066x (phase 44's
committed values -- identical construction, since 33h's floor was itself
measured store-mode). The rp floors are expected NEAR phase 44's 44.83 =
1.318x / 39.04 = 1.147x but not necessarily exact, and the difference is a
CONSTRUCTION fact worth recording: phases 43/44 priced the rp layout on
the UNCOMPRESSED arm's accuracy, while this phase prices the arm AS
DEPLOYED (store-mode accuracy). Drift up to the store-mode delta (|dACC|
<= 0.005, i.e. ~+-0.5 KB/pt) is attributable; anything larger is a
finding.

P2 -- THE TRANSFER QUESTION (charter clause a). We EXPECT phase 43's bound
to transfer to the ladder: held-out mean |dACC(rp - c64)| <= 0.005 at both
K and both decoders, and held-out mean |dFORG(rp - c64)| <= 0.010 at both
K. Grounds: the layout's loss is exactly sqrt(R) per slot (harness section
15 pins the identity) ~ 1.6% relative L2 at the measured median R =
2.55e-04 against capture margins at ~0.99 overlap, and the per-slot phase
is GAUGE for every |.| consumer -- there is no mechanism pathway from this
quantization to the readout above its own noise floor; the FORG band is
set at 2x the ACC band because FORG is a difference of two ACC-like
numbers.
  THE LINE THAT KILLS (A), named in advance: a held-out mean dFORG >=
  +0.020 at either K (or held-out mean dACC <= -0.020) makes (A) NOT WORTH
  TAKING AT ANY COST SAVING. Defense in one sentence: 0.02 is the
  project's standing pre-registered claim threshold (33e), and on this
  ladder +0.02 FORG would roughly double the measured forgetting at K=112
  (probe 0.027) and triple it at K=160 (probe 0.012) -- destroying the
  retention axis, which is the capability the re-scoped gate stands on and
  the only reason a cheaper episodic store is worth having.
  THE MIDDLE BAND, ruled in advance: 0.010 < held-out mean |dFORG| < 0.020
  (or 0.005 < |dACC| < 0.020) is a REAL cost below claim threshold; the
  recommendation must then weigh it against the ~0.58x cost factor
  explicitly rather than calling (A) free.

P3 -- THE FORK'S RELATIVE PRICE IS K-INDEPENDENT (derived before
measuring, 33h's process rule). The rp/c64 KB/ACC-pt ratio at equal ACC is
(xi/2 + phase + graph + meta) / (xi + graph + meta). At K=112 with 33h's
measured terms: 44.83/76.10 = 0.589. At K=160, assuming all 160 slots live
(33d measured 160/160) and the graph term at this protocol's observation
count ~12.6 KB: (41.60 + 12.6 + 1.28)/(81.92 + 12.6 + 1.28) = 0.579.
PREDICTION: measured ratio 0.58 +- 0.02 at both K -- the (A) discount
neither grows nor shrinks with capacity, because it lives entirely in the
xi term.
  MUNDANE ACCOUNT (M3): the ratio is arithmetic over four byte terms three
  of which are shared, so this prediction can only miss if the ARMS
  diverge (different live counts or graph sizes between c64 and rp
  training runs) -- which would itself be a store-mode behavioral finding,
  reported as such.

P4 -- REPLAY RESEEDED. Held-out mean ACC 0.91 +- 0.02, FORG 0.105 +-
0.03 (33c's committed single-seed 0.913/0.105; probe s0 0.9100/0.1058).
OPEN QUESTION registered with its probe value disclosed: at K=160 the
calib-b8 organism arm's probe (0.9301, seed 0, c64) sits ABOVE replay's
probe (0.9100). Whether organism-above-replay holds on held-out seeds is
measured here, not assumed; if it does, it is the first organism
configuration to top the ladder's best gradient arm on ACC and it MUST be
quoted with its cost sentence (K=160 organism ~55-96 KB depending on
branch, vs replay's 89.6 KB which additionally stores 200 raw labeled
samples) and with the standing caveat that the bar (0.872) is still above
the K=112 arm.

M1 -- THE NAMED MUNDANE ACCOUNT (charter clause b). The bound transfers
fine and the whole exercise merely re-derives phase 43 on a second
protocol: dACC/dFORG inside the P2 bands, floors inside P1/P3, nothing
new. That is a LEGITIMATE AND USEFUL outcome to record -- it converts the
fork's open behavioral question into a measured one on the protocol where
the organism is actually evaluated, which is exactly what the owner asked
for. If M1 is the outcome, the deliverable is the recommendation plus the
closed question, not a discovery.

HONEST-NEGATIVE BRANCH (charter clause c). If (A) costs real FORG (the P2
kill line: held-out mean dFORG >= +0.020 at either K), the fork resolves
to (B), the cost floor stands at 70.30 KB/ACC-pt = 2.066x (narrow CSR),
and the re-scoped gate absorbs it -- the 2026-08-11 decision record
already concedes the cost branch, so (B) costs a number the project has
already published, not a release.

SCOPING NOTE THE RECOMMENDATION MUST CARRY (charter). T7.6's arms (c)
dual time constant and (d) graph order are FORK-INDEPENDENT by T7.6's own
text, so branch (A) forfeits only arms (a) content-dependent phase and
(b) phase-aware readout -- and those two carry phase 43's measured
R ~ 0.06 readout-blindness bar and phase 16's measured 9-56-step
superposition collapse. The fork text as written ("permanently forfeit
phase-16 binding including the path-encoding T7.5/T7.6 would need")
OVERSTATES what (A) forfeits: the cheapest measured route to sequence
memory (T7.6c, a second O(N) field state, ~0.001x of the store) survives
branch (A) untouched.

WHAT THIS PHASE DOES NOT DO. It does not touch phase 48 or any T7.6 arm;
it does not change any default (`real_phase` stays default-OFF whatever
the recommendation says -- flipping it is the owner's act, not this
script's); and it RECOMMENDS rather than decides.

RUN COMMANDS (each saves its own stdout; `report` needs the others' cache):
    python phase50_fork_ladder.py --run bar     > phase50_results_bar.txt
    python phase50_fork_ladder.py --run k112    > phase50_results_K112.txt
    python phase50_fork_ladder.py --run k160    > phase50_results_K160.txt
    python phase50_fork_ladder.py --run replay  > phase50_results_replay.txt
    python phase50_fork_ladder.py --run report  > phase50_results_report.txt
Cells are cached in phase50_cells.json (scalars only, merged per run) so a
crash costs one part, not the day. Every cached number is also printed by
the run that measured it -- the report introduces no number the raw run
files cannot be checked against.
"""

import argparse
import json
import os
import time

import numpy as np

import phase33e_readout_geometry as p33e
import phase33h_cost_frontier as p33h
from organism_compress import CompressionSpec
from phase33c_gate_retest import N, TASKS

SEEDS = p33h.SEEDS                       # (0, 1, 2, 3, 4) selection
SEEDS_HELD = p33h.SEEDS_HELD             # (5, 6, 7, 8, 9) held out
ALL_SEEDS = SEEDS + SEEDS_HELD
KS = (112, 160)

DECODERS = (("argmax", dict(decoder="argmax")),
            ("calib-b8", dict(decoder="calib", m=None, beta=8.0)))

# the four store specs (43/44's exact objects, restated locally so this
# phase does not import phase43/phase44 module-level state)
SPEC_C64 = p33h.SPEC                     # c64 + CSR floor 0 + meta32
SPEC_C64N = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0,
                            meta_dtype=np.float32, p_narrow=True)
SPEC_RP = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0,
                          meta_dtype=np.float32, real_phase=True)
SPEC_RPN = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0,
                           meta_dtype=np.float32, real_phase=True,
                           p_narrow=True)
ARM_SPECS = {"c64": SPEC_C64, "rp": SPEC_RP}
NARROW_OF = {"c64": SPEC_C64N, "rp": SPEC_RPN}

CACHE = "phase50_cells.json"
REPLAY_BYTES_CONVENTION = ("params*4 + float32 buffer samples; labels and "
                           "optimizer state excluded -- 33h's 89.6KB "
                           "convention")

# pre-registered bands (P2) -- read by the report section, so they live in
# code as well as prose
BAND_DACC = 0.005
BAND_DFORG = 0.010
KILL_DFORG = 0.020
KILL_DACC = -0.020
CLAIM_THRESH = 0.02                      # 33e's standing claim threshold


# ---------------------------------------------------------------------------
# cache plumbing
# ---------------------------------------------------------------------------

def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(c):
    with open(CACHE, "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------

def frames_fed(seed):
    """Exact frame count for one full ladder run: sum over tasks of
    (task samples x epochs x hold). The observation count phase 44's rule
    wants attached to every byte number."""
    _, ytr, _, _ = p33e.build_split(seed)
    return int(sum(int(np.isin(ytr, t).sum()) * 3 * 8 for t in TASKS))


def run_organism_cell(K, arm, seed):
    """One (K, arm, seed) cell: 33h's run_arm verbatim, store mode, both
    decoders; byte accounting under the arm's own layout (CSR and narrow
    CSR) with observation counts."""
    t0 = time.time()
    res = p33h.run_arm(K, seed=seed, spec=ARM_SPECS[arm], decoders=DECODERS)
    org = res["org"]
    view = p33h.StoreView(org)
    nnz = int((np.asarray(view.P) > 0).sum())
    p_sum = float(np.asarray(view.P).sum())
    live = int(res["live"])
    cell = dict(
        seed=seed, K=K, arm=arm, live=live,
        acc_argmax=res["per_decoder"]["argmax"]["acc"],
        forg_argmax=res["per_decoder"]["argmax"]["forg"],
        acc_calib=res["per_decoder"]["calib-b8"]["acc"],
        forg_calib=res["per_decoder"]["calib-b8"]["forg"],
        bytes_csr=int(p33h.live_bytes(org, spec=ARM_SPECS[arm])),
        bytes_narrow=int(p33h.live_bytes(org, spec=NARROW_OF[arm])),
        nnz=nnz, p_sum=p_sum, density=nnz / max(live * live, 1),
        frames=frames_fed(seed), secs=round(time.time() - t0, 1))
    return cell


def run_replay_seed(seed, replay_per_class=20, epochs=300):
    """33c's `run_mlp('replay')` recipe, reseeded: torch.manual_seed(seed)
    and build_split(seed) replace the module-level seed-0 state; loop body
    otherwise verbatim (Adam 1e-3, 300 full-batch epochs/task, 20/class
    buffer)."""
    import torch
    import torch.nn as nn
    Xtr, ytr, Xte, yte = p33e.build_split(seed)
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(N, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    A = np.zeros((len(TASKS), len(TASKS)))
    buf_x, buf_y = [], []

    def pred(Xe):
        with torch.no_grad():
            return mlp(torch.tensor(Xe)).argmax(1).numpy()

    t0 = time.time()
    for ti, task in enumerate(TASKS):
        mask = np.isin(ytr, task)
        cur_x, cur_y = Xtr[mask], ytr[mask]
        if buf_x:
            cur_x = np.concatenate([cur_x] + buf_x)
            cur_y = np.concatenate([cur_y] + buf_y)
        xb = torch.tensor(cur_x); yb = torch.tensor(cur_y)
        for _ in range(epochs):
            opt.zero_grad(); lossf(mlp(xb), yb).backward(); opt.step()
        for c in task:
            idx = np.where(ytr[mask] == c)[0][:replay_per_class]
            buf_x.append(Xtr[mask][idx]); buf_y.append(ytr[mask][idx])
        A[ti] = p33e.eval_tasks(pred, Xte, yte)
    acc, forg = p33e.task_acc_matrix_to_metrics(A)
    params_b = sum(p.numel() for p in mlp.parameters()) * 4
    buf_b = int(sum(b.nbytes for b in buf_x))
    return dict(seed=seed, acc=acc, forg=forg, bytes=params_b + buf_b,
                params_bytes=params_b, buffer_bytes=buf_b,
                secs=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def boot_ci(deltas, B=20000, seed=7):
    """Percentile bootstrap CI of the mean of a small paired sample. n=5 is
    what the protocol gives; the CI is honest about that width."""
    d = np.asarray(deltas, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(B, d.size), replace=True).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def fmt_ci(deltas):
    m, lo, hi = boot_ci(deltas)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


# ---------------------------------------------------------------------------
# run sections
# ---------------------------------------------------------------------------

def do_bar(cache):
    print("=" * 70)
    print("(BAR) supervised prototype bar, RECOMPUTED ON EACH SEED'S OWN SPLIT")
    cells = {}
    for s in ALL_SEEDS:
        b = p33h.run_prototypes_seed(s)
        cells[str(s)] = dict(seed=s, acc=b["acc"], forg=b["forg"],
                             bytes=b["bytes"], n=b["n"])
        tag = "held" if s in SEEDS_HELD else "sel "
        print(f"  s{s} {tag}  ACC {b['acc']:.4f}  FORG {b['forg']:.4f}  "
              f"{b['n']:3d} protos  {b['bytes'] / 1e3:.2f} KB  "
              f"{p33h.kbpp(b['bytes'], b['acc']):.2f} KB/ACC-pt")
    a0 = cells["0"]
    ok = abs(a0["acc"] - 0.872) < 5e-4 and a0["bytes"] == 30720 and a0["n"] == 120
    print(f"  seed-0 anchor (0.872 / 120 protos / 30720 B): "
          f"{'REPRODUCED' if ok else 'MISMATCH'}")
    cache["bar"] = cells
    return cache


def do_k(cache, K):
    print("=" * 70)
    print(f"(K={K}) both arms, store mode, both decoders, seeds 0-9")
    print(f"  arm specs: c64 = {SPEC_C64.label()}   rp = {SPEC_RP.label()}")
    cells = cache.get(f"K{K}", {})
    for arm in ("c64", "rp"):
        cells.setdefault(arm, {})
        for s in ALL_SEEDS:
            if str(s) in cells[arm]:
                continue
            c = run_organism_cell(K, arm, s)
            cells[arm][str(s)] = c
            tag = "held" if s in SEEDS_HELD else "sel "
            print(f"  {arm:3s} s{s} {tag} live {c['live']:3d}  "
                  f"argmax {c['acc_argmax']:.4f}/{c['forg_argmax']:.4f}  "
                  f"calib-b8 {c['acc_calib']:.4f}/{c['forg_calib']:.4f}  "
                  f"csr {c['bytes_csr'] / 1e3:6.2f}KB narrow "
                  f"{c['bytes_narrow'] / 1e3:6.2f}KB  nnz {c['nnz']:5d} "
                  f"sum(P) {c['p_sum']:9.0f} dens {c['density']:.3f}  "
                  f"[{c['secs']}s]")
            cache[f"K{K}"] = cells
            save_cache(cache)
    # paired per-seed deltas, printed by the run that measured them
    print(f"\n  paired deltas rp - c64 at K={K} (per seed):")
    for dec in ("argmax", "calib"):
        for met in ("acc", "forg"):
            key = f"{met}_{dec}"
            dsel = [cells["rp"][str(s)][key] - cells["c64"][str(s)][key]
                    for s in SEEDS]
            dheld = [cells["rp"][str(s)][key] - cells["c64"][str(s)][key]
                     for s in SEEDS_HELD]
            print(f"    d{met.upper()} {dec:8s} sel  {fmt_ci(dsel)}   "
                  f"held {fmt_ci(dheld)}")
    cache[f"K{K}"] = cells
    return cache


def do_replay(cache):
    print("=" * 70)
    print("(REPLAY) experience replay reseeded per split "
          "(33c recipe, torch.manual_seed(seed))")
    import torch
    print(f"  torch {torch.__version__} (33c's committed run was 2.13-cpu "
          f"at module-level seed 0 after seq/ewc arms consumed rng state; "
          f"deltas vs 0.913/0.105 are rng-provenance, not protocol)")
    cells = cache.get("replay", {})
    for s in ALL_SEEDS:
        if str(s) in cells:
            continue
        c = run_replay_seed(s)
        cells[str(s)] = c
        tag = "held" if s in SEEDS_HELD else "sel "
        print(f"  s{s} {tag}  ACC {c['acc']:.4f}  FORG {c['forg']:.4f}  "
              f"{c['bytes'] / 1e3:.1f} KB ({c['params_bytes'] / 1e3:.1f} params "
              f"+ {c['buffer_bytes'] / 1e3:.1f} buffer)  [{c['secs']}s]")
        cache["replay"] = cells
        save_cache(cache)
    held = [cells[str(s)]["acc"] for s in SEEDS_HELD]
    heldf = [cells[str(s)]["forg"] for s in SEEDS_HELD]
    print(f"  held-out mean ACC {np.mean(held):.4f}  FORG {np.mean(heldf):.4f}  "
          f"(bytes convention: {REPLAY_BYTES_CONVENTION})")
    cache["replay"] = cells
    return cache


# ---------------------------------------------------------------------------
# the report -- tables, pre-registration outcomes, recommendation
# ---------------------------------------------------------------------------

def _arm_stats(cells, arm, seeds, key):
    return np.array([cells[arm][str(s)][key] for s in seeds])


def do_report(cache):
    for need in ("bar", "K112", "K160", "replay"):
        if need not in cache:
            raise SystemExit(f"cache is missing '{need}' -- run it first")
    bar, replay = cache["bar"], cache["replay"]

    print("=" * 70)
    print("PHASE 50 (T7.7) REPORT -- the fork measured on the ladder")
    print("=" * 70)

    # ---------------- ladder table --------------------------------------
    print("\n(1) LADDER TABLE -- mean over seeds, sel = 0-4, held = 5-9")
    print(f"    {'arm':24s} {'ACC sel':>9} {'ACC held':>9} "
          f"{'FORG sel':>9} {'FORG held':>10}")
    for K in KS:
        cells = cache[f"K{K}"]
        for arm in ("c64", "rp"):
            for dec in ("argmax", "calib"):
                a_s = _arm_stats(cells, arm, SEEDS, f"acc_{dec}").mean()
                a_h = _arm_stats(cells, arm, SEEDS_HELD, f"acc_{dec}").mean()
                f_s = _arm_stats(cells, arm, SEEDS, f"forg_{dec}").mean()
                f_h = _arm_stats(cells, arm, SEEDS_HELD, f"forg_{dec}").mean()
                print(f"    K={K} {arm:4s} {dec:8s}        "
                      f"{a_s:9.4f} {a_h:9.4f} {f_s:9.4f} {f_h:10.4f}")
    for name, cells_ in (("prototype bar", bar), ("replay", replay)):
        a_s = np.mean([cells_[str(s)]["acc"] for s in SEEDS])
        a_h = np.mean([cells_[str(s)]["acc"] for s in SEEDS_HELD])
        f_s = np.mean([cells_[str(s)]["forg"] for s in SEEDS])
        f_h = np.mean([cells_[str(s)]["forg"] for s in SEEDS_HELD])
        print(f"    {name:24s}{a_s:9.4f} {a_h:9.4f} {f_s:9.4f} {f_h:10.4f}")
    print("    scope: bar and replay are K-independent (no slot bank); the "
          "bar is supervised, replay stores 200 raw labeled samples.")

    # ---------------- paired deltas with CIs ------------------------------
    print("\n(2) PAIRED FORK DELTAS rp - c64, bootstrap 95% CIs (B=20000)")
    deltas = {}
    for K in KS:
        cells = cache[f"K{K}"]
        for dec in ("argmax", "calib"):
            for met in ("acc", "forg"):
                key = f"{met}_{dec}"
                dsel = [cells["rp"][str(s)][key] - cells["c64"][str(s)][key]
                        for s in SEEDS]
                dheld = [cells["rp"][str(s)][key] - cells["c64"][str(s)][key]
                         for s in SEEDS_HELD]
                deltas[(K, dec, met)] = (dsel, dheld)
                print(f"    K={K} {dec:8s} d{met.upper():4s}  "
                      f"sel {fmt_ci(dsel)}   held {fmt_ci(dheld)}")

    # ---------------- cost table ------------------------------------------
    print("\n(3) KB/ACC-pt, HELD-OUT seeds (33h's metric: per-seed KB/acc, "
          "then mean; bar recomputed per seed)")
    bar_k = np.mean([p33h.kbpp(bar[str(s)]["bytes"], bar[str(s)]["acc"])
                     for s in SEEDS_HELD])
    print(f"    bar held-out: {bar_k:.2f} KB/ACC-pt "
          f"(ACC {np.mean([bar[str(s)]['acc'] for s in SEEDS_HELD]):.4f}, "
          f"{bar['5']['n']}-{bar['9']['n']} protos across seeds)")
    floors = {}
    for K in KS:
        cells = cache[f"K{K}"]
        for arm in ("c64", "rp"):
            for enc, bkey in (("CSR", "bytes_csr"), ("narrowCSR", "bytes_narrow")):
                ks = [p33h.kbpp(cells[arm][str(s)][bkey],
                                cells[arm][str(s)]["acc_calib"])
                      for s in SEEDS_HELD]
                floors[(K, arm, enc)] = (float(np.mean(ks)),
                                         float(np.mean(ks)) / bar_k)
        nnz_h = _arm_stats(cells, "c64", SEEDS_HELD, "nnz")
        psum_h = _arm_stats(cells, "c64", SEEDS_HELD, "p_sum")
        for arm in ("c64", "rp"):
            for enc in ("CSR", "narrowCSR"):
                kb, ratio = floors[(K, arm, enc)]
                b = np.mean(_arm_stats(cache[f"K{K}"], arm, SEEDS_HELD,
                                       "bytes_csr" if enc == "CSR"
                                       else "bytes_narrow")) / 1e3
                print(f"    K={K} {arm:4s}+{enc:9s} {b:6.2f} KB  "
                      f"{kb:6.2f} KB/ACC-pt = {ratio:.3f}x the bar")
        print(f"      observation counts at K={K} (phase 44's rule; c64 arm, "
              f"held-out means): frames "
              f"{np.mean(_arm_stats(cells, 'c64', SEEDS_HELD, 'frames')):.0f}, "
              f"sum(P) {psum_h.mean():.0f}, nnz {nnz_h.mean():.0f}, density "
              f"{np.mean(_arm_stats(cells, 'c64', SEEDS_HELD, 'density')):.3f} "
              f"over live "
              f"{np.mean(_arm_stats(cells, 'c64', SEEDS_HELD, 'live')):.0f} "
              f"slots; rp arm nnz "
              f"{np.mean(_arm_stats(cells, 'rp', SEEDS_HELD, 'nnz')):.0f}")
    rep_b = np.mean([replay[str(s)]["bytes"] for s in SEEDS_HELD])
    rep_a = np.mean([replay[str(s)]["acc"] for s in SEEDS_HELD])
    rep_k = np.mean([p33h.kbpp(replay[str(s)]["bytes"], replay[str(s)]["acc"])
                     for s in SEEDS_HELD])
    print(f"    replay held-out: {rep_b / 1e3:.1f} KB at ACC {rep_a:.4f} = "
          f"{rep_k:.2f} KB/ACC-pt = {rep_k / bar_k:.3f}x the bar "
          f"({REPLAY_BYTES_CONVENTION})")
    print("    scope: KB/ACC-pt at UNEQUAL accuracy answers a narrower "
          "question than the gate's cost branch did; the branch is conceded "
          "(2026-08-11) and these are architecture-choice numbers.")

    # ---------------- pre-registration outcomes ---------------------------
    print("\n(4) PRE-REGISTRATION OUTCOMES")
    k112 = cache["K112"]
    # P1 anchors
    bar0_ok = (abs(bar["0"]["acc"] - 0.872) < 5e-4 and bar["0"]["bytes"] == 30720)
    k160_s0 = cache["K160"]["c64"]["0"]["acc_argmax"]
    k160_ok = abs(k160_s0 - 0.900) < 5e-4
    f_c64 = floors[(112, "c64", "CSR")]
    f_c64n = floors[(112, "c64", "narrowCSR")]
    f_rp = floors[(112, "rp", "CSR")]
    f_rpn = floors[(112, "rp", "narrowCSR")]
    p1_c64 = abs(f_c64[0] - 76.10) <= 0.30 and abs(f_c64[1] - 2.236) <= 0.010
    p1_c64n = abs(f_c64n[0] - 70.30) <= 0.30 and abs(f_c64n[1] - 2.066) <= 0.010
    p1_rp = abs(f_rp[0] - 44.83) <= 0.50
    p1_rpn = abs(f_rpn[0] - 39.04) <= 0.50
    print(f"    P1 bar anchor {'REPRODUCED' if bar0_ok else 'MISMATCH'}; "
          f"K=160 c64 argmax s0 {k160_s0:.4f} vs 0.900 "
          f"{'REPRODUCED' if k160_ok else 'MISMATCH'}")
    print(f"       K=112 floors: c64 {f_c64[0]:.2f}={f_c64[1]:.3f}x "
          f"[{'ok' if p1_c64 else 'DRIFT'} vs 76.10=2.236x]  "
          f"c64n {f_c64n[0]:.2f}={f_c64n[1]:.3f}x "
          f"[{'ok' if p1_c64n else 'DRIFT'} vs 70.30=2.066x]")
    print(f"       rp {f_rp[0]:.2f}={f_rp[1]:.3f}x "
          f"[{'ok' if p1_rp else 'DRIFT'} vs 44.83, +-0.5 attributable]  "
          f"rpn {f_rpn[0]:.2f}={f_rpn[1]:.3f}x "
          f"[{'ok' if p1_rpn else 'DRIFT'} vs 39.04]")
    # P2 transfer + kill line
    worst_dacc, worst_dforg, kill = 0.0, 0.0, False
    p2_ok = True
    for K in KS:
        for dec in ("argmax", "calib"):
            dh_a = np.mean(deltas[(K, dec, "acc")][1])
            dh_f = np.mean(deltas[(K, dec, "forg")][1])
            worst_dacc = min(worst_dacc, dh_a)
            worst_dforg = max(worst_dforg, dh_f)
            if abs(dh_a) > BAND_DACC or abs(dh_f) > BAND_DFORG:
                p2_ok = False
            if dh_f >= KILL_DFORG or dh_a <= KILL_DACC:
                kill = True
    print(f"    P2 transfer ({'HELD' if p2_ok else 'MISSED'}): worst held-out "
          f"mean dACC {worst_dacc:+.4f} (band +-{BAND_DACC}), worst dFORG "
          f"{worst_dforg:+.4f} (band +-{BAND_DFORG})")
    print(f"       kill line (dFORG >= +{KILL_DFORG} or dACC <= {KILL_DACC}): "
          f"{'TRIGGERED -- (A) is dead at any saving' if kill else 'not triggered'}")
    # P3 ratio
    p3_all = True
    for K in KS:
        r = floors[(K, "rp", "CSR")][0] / floors[(K, "c64", "CSR")][0]
        ok = abs(r - 0.58) <= 0.02
        p3_all &= ok
        print(f"    P3 rp/c64 price ratio at K={K}: {r:.3f} "
              f"[{'ok' if ok else 'MISS'} vs 0.58 +- 0.02]")
    # P4 replay
    p4_ok = abs(rep_a - 0.91) <= 0.02
    org160_h = _arm_stats(cache["K160"], "c64", SEEDS_HELD, "acc_calib")
    rep_h = np.array([replay[str(s)]["acc"] for s in SEEDS_HELD])
    above = org160_h - rep_h
    print(f"    P4 replay held-out ACC {rep_a:.4f} "
          f"[{'ok' if p4_ok else 'MISS'} vs 0.91 +- 0.02]; open question: "
          f"K=160 c64 calib-b8 minus replay, per held-out seed "
          f"{np.round(above, 4).tolist()} (mean {above.mean():+.4f}) -> "
          f"organism above replay on "
          f"{int((above > 0).sum())}/5 held-out seeds")

    # ---------------- recommendation --------------------------------------
    print("\n(5) RECOMMENDATION (decision-rule-driven; the owner acts, "
          "this script argues)")
    if kill:
        print("    The pre-registered kill line FIRED: (A) costs real "
              "forgetting on the ladder. RECOMMENDATION: resolve the fork "
              "to (B); the floor stands at "
              f"{f_c64n[0]:.2f} KB/ACC-pt = {f_c64n[1]:.3f}x (narrow CSR), "
              "which the re-scoped gate absorbs.")
    elif not p2_ok:
        print("    The bound did NOT transfer clean (middle band): a real "
              "cost below claim threshold. RECOMMENDATION: weigh the "
              f"measured held-out costs (worst dACC {worst_dacc:+.4f}, "
              f"dFORG {worst_dforg:+.4f}) against the "
              f"{floors[(112, 'rp', 'CSR')][0] / floors[(112, 'c64', 'CSR')][0]:.2f}x "
              "cost factor -- (A) is cheaper but not free; the owner's "
              "tolerance on the retention axis decides.")
    else:
        print("    M1 (the named mundane account) is the outcome: phase 43's "
              "bound TRANSFERS to the ladder -- the (A) store is behavior-"
              "neutral where the organism is actually evaluated, on both "
              "axes, both K, held out. RECOMMENDATION: take branch (A) "
              "(real_phase store, with T7.2's narrow CSR) as the DEPLOYMENT "
              "persistence layout -- "
              f"{f_rpn[0]:.2f} KB/ACC-pt = {f_rpn[1]:.3f}x the bar vs "
              f"{f_c64n[0]:.2f} = {f_c64n[1]:.3f}x for (B) -- while keeping "
              "complex128 as the COMPUTE width (unchanged E2 contract) and "
              "real_phase default-OFF in code until the owner flips it.")
    print("    Scoping note the charter mandates: T7.6 arms (c) dual time "
          "constant and (d) graph order are fork-independent, so (A) "
          "forfeits ONLY arms (a)+(b) -- which carry phase 43's R ~ 0.06 "
          "readout-blindness bar and phase 16's 9-56-step superposition "
          "collapse. The fork text as written overstates what (A) forfeits: "
          "the cheapest measured route to sequence memory (T7.6c) survives "
          "(A) untouched. Reversibility is asymmetric and belongs in the "
          "decision: (B) can move to (A) later by re-compressing stored "
          "state, while (A) back to (B) loses whatever the imaginary "
          "channel would have accumulated -- but phase 43 measured that "
          "nothing accumulates at the committed parameters (median R "
          "2.6e-04) and no committed pipeline feeds complex input, so what "
          "(A) actually spends today is an OPTION, priced by the R ~ 0.06 "
          "bar T7.6(a)/(b) would have to clear anyway.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    choices=["bar", "k112", "k160", "replay", "report", "all"])
    args = ap.parse_args()
    t0 = time.time()
    print("PHASE 50 (T7.7): fork decision measurement -- the (A) store on "
          "the ladder")
    print(f"  run={args.run}  seeds sel={list(SEEDS)} held={list(SEEDS_HELD)}"
          f"  K={list(KS)}  decoders={[d for d, _ in DECODERS]}")
    cache = load_cache()
    if args.run in ("bar", "all"):
        cache = do_bar(cache); save_cache(cache)
    if args.run in ("k112", "all"):
        cache = do_k(cache, 112); save_cache(cache)
    if args.run in ("k160", "all"):
        cache = do_k(cache, 160); save_cache(cache)
    if args.run in ("replay", "all"):
        cache = do_replay(cache); save_cache(cache)
    if args.run in ("report", "all"):
        do_report(cache)
    print(f"\nTOTAL {time.time() - t0:.1f}s")
