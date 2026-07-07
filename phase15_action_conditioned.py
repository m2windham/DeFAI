"""
PHASE 15 -- ACTION-CONDITIONED ATTRACTOR TRANSITIONS: from observer to agent

Everything before this phase learns State -> State transitions: a Markov
chain of observations. An agent needs State x Action -> State: an efference
copy (the action it intends) selects WHICH transition prior applies. The
action is not an output of an attractor; it is a parameter that changes
which attractors are reachable next.

Minimal grounding testbed: a 4x4 gridworld, 16 states with random
embeddings, 4 actions (N/S/E/W, walls block: the move fails and the agent
stays). The organism perceives ONLY state embeddings from its own random
walk plus its own action labels (efference copy -- observable by
definition). No coordinates, no reward, no map. Slots recruited online as
always; per-action Hebbian counts once per step (occurrence-level, the
phase-8 lesson).

Tests (the ground-truth grid is used only for scoring):
  1. FORWARD MODEL: given (state, action), predict the next state from the
     learned per-action counts. Baseline: the action-BLIND marginal P used
     by every prior phase. If conditioning matters, the gap is the point.
  2. NAVIGATION BY IMAGINATION: random start, random goal. The agent plans
     ENTIRELY inside its learned model -- breadth-first search over the
     imagined graph (slot --action--> predicted slot), executes the first
     action, replans each step. Success = reach the goal within a horizon.
     Baselines: random actions; action-blind agent (no plan possible).
  3. MULTI-STEP IMAGINATION: roll the forward model 5 actions deep with no
     input and score every imagined waypoint against the true trajectory.

RESULT (recorded from the committed run):
  - Forward model 1.000 over all 64 (state, action) pairs vs 0.312 for the
    action-blind marginal: the efference copy is the entire difference
    between a world model and a Markov chain of observations.
  - Navigation by planning INSIDE the learned model: 1.000 success over 300
    episodes (random actions: 0.340). The agent has never seen a map or a
    coordinate; it navigates by imagining.
  - Imagination rollouts 5 actions deep: 1.000 of 1000 waypoints correct.
  - Method note, preserved because it is instructive: the first run scored
    at the action-blind baseline (0.281) due to an efference-copy off-by-one
    -- transitions were recorded with the action about to be taken instead
    of the action that caused the arrival. In an oscillator agent the
    efference copy must be time-aligned with the CONSEQUENCE, not the
    intention.
"""

import numpy as np
from organism import normalize

SIDE = 4
N_STATES = SIDE * SIDE
ACTIONS = ['N', 'S', 'E', 'W']
DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

DIM = 48; NORM = np.sqrt(DIM)
HOLD = 8


def step_env(s, a):
    r, c = divmod(s, SIDE)
    dr, dc = DELTA[a]
    r2, c2 = r + dr, c + dc
    if 0 <= r2 < SIDE and 0 <= c2 < SIDE:
        return r2 * SIDE + c2
    return s


emb_rng = np.random.default_rng(7)
emb = emb_rng.standard_normal((N_STATES, DIM))
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
states = np.array([normalize(emb[s].astype(complex), NORM) for s in range(N_STATES)])


class ActionOrg:
    """Slot memory with per-action transition counts. Perception is the
    usual settle-and-gate; the only new machinery is that Hebbian counts
    are indexed by the agent's own efference copy."""

    def __init__(self, N, K, n_actions, omega=0.15, seed=0):
        self.N = N; self.K = K
        self.norm = np.sqrt(N)
        self.omega = omega
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pa = np.zeros((n_actions, K, K))   # action-conditioned counts
        self.Pm = np.zeros((K, K))              # action-blind marginal (baseline)
        self.z = normalize(self.rng.standard_normal(N).astype(complex), self.norm)
        self.prev_k = -1

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def observe(self, x, g_in=5.0, dt=0.05, recruit=0.5, eta=0.05):
        z = self.z
        for _ in range(HOLD * 8):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        self.z = z
        used_idx = np.where(self.used)[0]
        if len(used_idx) > 0:
            ovs = np.abs(self.overlaps(z, self.xi[used_idx]))
            k = int(used_idx[np.argmax(ovs)]); best = ovs.max()
        else:
            k = -1; best = 0.0
        if best < recruit and not self.used.all():
            k = int(np.argmin(self.used.astype(float)))
            self.xi[k] = z.copy(); self.used[k] = True
        else:
            ph = np.exp(-1j*np.angle(self.overlaps(z, self.xi[[k]])[0]))
            self.xi[k] = normalize(self.xi[k] + eta*(z*ph - self.xi[k]), self.norm)
        self.count[k] += 1
        return k

    def learn_step(self, x, action_into):
        """action_into: the efference copy of the action that LED TO this
        observation -- the transition being learned is (prev --action--> here).
        Passing the action about to be taken instead silently fills every
        per-action table with other actions' transitions (measured: forward
        model drops to the action-blind baseline)."""
        k = self.observe(x)
        if self.prev_k >= 0 and action_into is not None:
            self.Pa[action_into][self.prev_k, k] += 1
            self.Pm[self.prev_k, k] += 1
        self.prev_k = k
        return k

    def predicted_next(self, k, a):
        row = self.Pa[a][k]
        return int(np.argmax(row)) if row.sum() > 0 else -1

    def plan(self, k_start, k_goal, max_depth=12):
        """BFS entirely inside the learned model: slots are nodes, the
        imagined outcome of each action is an edge. Returns first action of
        the shortest imagined path, or None."""
        if k_start == k_goal:
            return None
        frontier = [(k_start, [])]
        seen = {k_start}
        while frontier:
            node, path = frontier.pop(0)
            if len(path) >= max_depth:
                continue
            for a in range(len(ACTIONS)):
                nxt = self.predicted_next(node, a)
                if nxt < 0 or nxt in seen:
                    continue
                if nxt == k_goal:
                    return (path + [a])[0]
                seen.add(nxt)
                frontier.append((nxt, path + [a]))
        return None


# ---- learn from a random walk --------------------------------------------------
WALK = 4000
rng = np.random.default_rng(0)
org = ActionOrg(N=DIM, K=24, n_actions=4, seed=0)
s = 0
a_prev = None
truth_pairs = []
for t in range(WALK):
    org.learn_step(states[s], a_prev)
    a = int(rng.integers(4))
    s2 = step_env(s, a)
    truth_pairs.append((s, a, s2))
    a_prev = a; s = s2
org.learn_step(states[s], a_prev)   # observe the final arrival too
# (state, action) pairs whose transition was learned, for honest scoring
seen_sa = set((s0, a) for s0, a, _ in truth_pairs)

# slot <-> state maps (evaluation only)
used_idx = np.where(org.used)[0]
ov = np.abs((org.xi[used_idx].conj() @ states.T) / DIM)
s2slot = {st: int(used_idx[ov[:, st].argmax()]) for st in range(N_STATES)}
slot2s = {int(used_idx[i]): int(ov[i].argmax()) for i in range(len(used_idx))}
coverage = np.mean([slot2s[s2slot[st]] == st for st in range(N_STATES)])
print(f"random walk: {WALK} steps  slots used: {len(used_idx)}/24  "
      f"state coverage: {coverage:.2f}")

# ---- 1. forward model ----------------------------------------------------------
ok_a = ok_m = tot = 0
for st in range(N_STATES):
    for a in range(4):
        if (st, a) not in seen_sa:
            continue
        true_next = step_env(st, a)
        k = s2slot[st]
        pa = org.predicted_next(k, a)
        ok_a += pa >= 0 and slot2s.get(pa) == true_next
        pm = int(np.argmax(org.Pm[k])) if org.Pm[k].sum() > 0 else -1
        ok_m += pm >= 0 and slot2s.get(pm) == true_next
        tot += 1
fm_acc = ok_a / max(tot, 1); fm_blind = ok_m / max(tot, 1)
print(f"\n1. FORWARD MODEL over {tot} seen (state, action) pairs")
print(f"   action-conditioned: {fm_acc:.3f}    action-blind baseline: {fm_blind:.3f}")

# ---- 2. navigation by imagination ----------------------------------------------
def episode(policy, horizon=10, seed=0):
    local = np.random.default_rng(seed)
    s = int(local.integers(N_STATES))
    g = int(local.integers(N_STATES))
    while g == s:
        g = int(local.integers(N_STATES))
    for t in range(horizon):
        a = policy(s, g, local)
        if a is None:
            return s == g
        s = step_env(s, a)
        if s == g:
            return True
    return False

def planner_policy(s, g, local):
    a = org.plan(s2slot[s], s2slot[g])
    return a if a is not None else int(local.integers(4))

def random_policy(s, g, local):
    return int(local.integers(4))

EPS = 300
nav_plan = np.mean([episode(planner_policy, seed=i) for i in range(EPS)])
nav_rand = np.mean([episode(random_policy, seed=i) for i in range(EPS)])
print(f"\n2. NAVIGATION ({EPS} episodes, random start/goal, horizon 10)")
print(f"   planning in imagination: {nav_plan:.3f}    random actions: {nav_rand:.3f}")

# ---- 3. multi-step imagination -------------------------------------------------
DEPTH = 5
ok = tot = 0
for trial in range(200):
    local = np.random.default_rng(1000 + trial)
    s = int(local.integers(N_STATES))
    acts = [int(local.integers(4)) for _ in range(DEPTH)]
    k = s2slot[s]
    for a in acts:
        s = step_env(s, a)
        k = org.predicted_next(k, a)
        if k < 0:
            break
        ok += slot2s.get(k) == s
        tot += 1
im_acc = ok / max(tot, 1)
print(f"\n3. IMAGINATION ROLLOUT (5 actions deep, no input, 200 trials)")
print(f"   imagined waypoints correct: {im_acc:.3f}  ({tot} waypoints scored)")

print("\nverdict:", "efference copy turns the transition prior into a usable world"
      " model -- the organism navigates by imagining"
      if fm_acc > 0.9 and nav_plan > nav_rand + 0.3 and im_acc > 0.8 else
      "partial -- see per-test numbers; the model or the planner is the gap")
