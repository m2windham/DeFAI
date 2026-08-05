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

    def predict(self, org, X):
        """Majority label of each query's best-overlapping evidenced slot.
        Reads org state; never writes it."""
        slots, lab = self.slot_labels(org)
        states = self._states(org, X)
        ov = np.abs(org.xi[slots].conj() @ states.T) / org.N
        return lab[np.argmax(ov, axis=0)]

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
