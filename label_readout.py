"""
LABEL-EVIDENCE READOUT -- EVAL-SIDE ONLY. Promoted out of
`phase33_industry_baselines.py` (roadmap target T1.3): the online per-slot
label-evidence readout that fixed phase 33's frozen-label artifact
(ACC 0.25 -> 0.665), now a reusable mechanism instead of an in-phase patch.

What it is
----------
A readout layer that sits OUTSIDE the organism and maps its unsupervised
memory slots to task labels by accumulating evidence online: after the
organism perceives a batch, each labeled sample votes for the slot whose
attractor it overlaps most, and the vote is recorded as
`evidence[slot, label] += 1`. Prediction assigns a query to its
best-overlapping evidenced slot and returns that slot's majority label.

Because evidence keeps accumulating batch by batch, a slot that DRIFTS
toward a later class simply out-votes its old label -- exactly the failure
a frozen (first-assignment) labeling mis-scores. Phase 33 measured that
difference on class-incremental split-digits: frozen labels score the
drift, not the memory (ACC 0.25); online evidence scores the memory
(ACC 0.665).

The label boundary (standing rule: no labels inside the mechanism)
------------------------------------------------------------------
Labels enter HERE and nowhere else. The readout only READS organism state
(`xi`, `used`, `norm`) and never writes any of it -- perception, slot
recruitment, Hebbian refinement, and the transition graph stay fully
unsupervised. `test_label_readout.py` pins this: `observe`/`predict` leave
every organism attribute bitwise unchanged. Anything that feeds labels
back into perception is out of scope by construction and would violate
the project's core premise.

Slot-lifecycle notifications
----------------------------
Evidence is indexed by slot, so slot-identity events must be forwarded,
mirroring `EventBoundary`'s contract in `organism.py` (which already emits
them for pool-mode fusion and use-it-or-lose-it recycling):

    readout.remap(drop, keep)     # slot fusion: drop's identity -> keep
    readout.invalidate(expired)   # recycled/evicted slots (bool mask or ints)

Phase 33 runs the organism in plain mode where slot indices are stable, so
its script needs neither. A future eviction policy (target T1.2) or any
pool-mode evaluation should wire the organism's remap/invalidate
notifications through to the readout at the phase-script level -- no hook
inside organism.py is required, the notifications already exist.

Persistence: `evidence` is a plain (K, n_classes) float array -- eval-side
state, deliberately NOT part of the E3 organism schema. Scripts that need
it across runs can np.save/np.load it alongside the E3 .npz.

Decoder geometry (T1.6)
-----------------------
`predict` accepts a `decoder` argument -- richer EVAL-SIDE decoders over
the same organism state and the same evidence matrix (labels still enter
only through `observe`; every decoder reads org state and never writes it):

    argmax  (default)  best-overlap evidenced slot -> its majority label.
                       The original phase-33 readout, byte-for-byte.
    soft               evidence-weighted soft vote: the top-m overlapping
                       slots each contribute overlap * evidence[slot] and
                       the class scores are summed before the argmax, so
                       label mass a majority collapse throws away counts.
    dist               same top-m routing, but each slot contributes its
                       label DISTRIBUTION (evidence row normalized to sum
                       1) so a high-count slot cannot outvote a purer
                       low-count one on raw mass alone.
    calib              distance-calibrated: softmax(beta * overlap) weights
                       over ALL evidenced slots (m=None) or the top-m,
                       times per-slot label distributions -- the continuous
                       version of top-m truncation. beta -> inf recovers
                       nearest-slot routing; beta = 0 is a uniform average.

All three reduce EXACTLY to `argmax` at m=1 (single slot: its weighted
row, normalized or not, argmaxes to the majority label) -- pinned in
`test_label_readout.py`. Measured on the phase-33c protocol by
`phase33e_readout_geometry.py`; `argmax` stays the default.
"""

import numpy as np

from organism import normalize


class LabelEvidenceReadout:
    """Online per-slot label evidence over an organism's memory slots.

    Typical loop (phase-33 shape)::

        readout = LabelEvidenceReadout(K=org.K, n_classes=10)
        for X_task, y_task in tasks:
            org.perceive(encode(X_task), ...)          # unsupervised, as ever
            readout.observe(org, X_task, y_task)       # labels enter here only
        y_hat = readout.predict(org, X_query)

    `X` may be raw real feature rows (encoded internally with the organism's
    own norm, matching what phase scripts feed `perceive`) or already-encoded
    complex states of shape (n, N) -- complex input is used as-is.
    """

    def __init__(self, K, n_classes):
        self.evidence = np.zeros((K, n_classes))

    # ---- encoding -------------------------------------------------------
    @staticmethod
    def _states(org, X):
        X = np.asarray(X)
        if np.iscomplexobj(X):
            return X
        return np.array([normalize(x.astype(complex), org.norm) for x in X])

    # ---- evidence accumulation (labels enter the system here) -----------
    def observe(self, org, X, labels):
        """Accumulate evidence from the CURRENT batch only -- no stored past
        data, no end-of-run oracle. Each sample votes for its nearest used
        slot (max |<xi_k, state>|); the vote is one count toward that slot's
        label. Reads org state; never writes it."""
        used = np.where(org.used)[0]
        labels = np.asarray(labels, int)
        if used.size == 0 or labels.size == 0:
            return
        states = self._states(org, X)
        ov = np.abs(org.xi[used].conj() @ states.T) / org.N
        near = used[np.argmax(ov, axis=0)]
        np.add.at(self.evidence, (near, labels), 1.0)

    # ---- prediction -----------------------------------------------------
    def evidenced(self, org):
        """Used slots that have received at least one vote -- the only slots
        prediction may route through."""
        used = np.where(org.used)[0]
        return used[self.evidence[used].sum(1) > 0]

    def slot_labels(self, org):
        """(slots, labels): each evidenced slot's current majority label.
        Diagnostic view -- watching this across tasks shows relabeling-by-
        drift directly (phase 33's slot-flooding measurement)."""
        slots = self.evidenced(org)
        return slots, self.evidence[slots].argmax(1)

    def predict(self, org, X, decoder="argmax", m=5, beta=16.0):
        """Predict labels for queries `X` from the accumulated evidence.
        Reads org state; never writes it.

        decoder="argmax" (default): majority label of each query's
        best-overlapping evidenced slot -- the original phase-33 readout,
        unchanged. The T1.6 alternatives (`soft`, `dist`, `calib`) are
        documented in the module docstring; `m` is the number of
        top-overlap slots consulted (None = all evidenced slots) and
        `beta` is the softmax sharpness for `calib`."""
        slots, lab = self.slot_labels(org)
        states = self._states(org, X)
        ov = np.abs(org.xi[slots].conj() @ states.T) / org.N   # (S, n)
        if decoder == "argmax":
            return lab[np.argmax(ov, axis=0)]
        if decoder not in ("soft", "dist", "calib"):
            raise ValueError(f"unknown decoder {decoder!r}")
        E = self.evidence[slots]                               # (S, C)
        if decoder in ("dist", "calib"):
            E = E / E.sum(1, keepdims=True)     # evidenced rows: sum > 0
        mm = len(slots) if m is None else min(int(m), len(slots))
        # stable sort so the top-1 slot on ties is argmax's first-max slot
        # (m=1 must reduce to the argmax decoder exactly)
        top = np.argsort(-ov, axis=0, kind="stable")[:mm]      # (m, n)
        w = np.take_along_axis(ov, top, axis=0)                # (m, n)
        if decoder == "calib":
            w = np.exp(beta * (w - w.max(0, keepdims=True)))
            w = w / w.sum(0, keepdims=True)
        return np.einsum("mn,mnc->nc", w, E[top]).argmax(1)

    # ---- slot-lifecycle notifications (EventBoundary contract) ----------
    def remap(self, drop, keep):
        """Slot fusion: `drop`'s identity continues as `keep` -- its votes
        fold into `keep` and `drop` starts clean."""
        self.evidence[keep] += self.evidence[drop]
        self.evidence[drop] = 0.0

    def invalidate(self, expired):
        """Recycled/evicted slots carry no evidence. `expired` is a boolean
        mask over slots or an index array, as EventBoundary receives it."""
        self.evidence[expired] = 0.0
