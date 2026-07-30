"""Batched PUCT MCTS.

Throughput, not per-tree cleverness, is what matters for AlphaZero self-play: the
GPU wants large inference batches, and a single game's search produces one leaf at
a time. So this searches ``G`` independent games in lockstep -- every simulation
round descends all ``G`` trees, collects their leaves into one batch, runs a single
network call, and backs up all ``G`` results. With ``G`` in the hundreds the GPU
stays saturated without needing virtual loss inside any one tree.

Tree storage is dense: ``[G, max_nodes, 140]`` arrays indexed directly by action.
That wastes memory on illegal actions but removes all ragged-offset bookkeeping
from the hot loop, and at 600 nodes x 140 actions x 4 bytes it is ~340 KB per game
per array -- cheap enough. Node rows are zeroed on creation rather than clearing
the whole arena each move.

Value sign convention: every value is from the perspective of the side to move at
that node, and flips once per ply on the way back up the path.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from quoridor.encoding import (
    NUM_PLANES,
    ROT180_ACTION,
    canonicalize,
    encode_canonical,
)
from quoridor.fastrules import (
    IDX_TURN,
    NUM_ACTIONS,
    SCRATCH_SIZE,
    STATE_SIZE,
    apply_action,
    legal_mask,
    winner,
)

MAX_DEPTH = 160


@njit(cache=True, parallel=True)
def _select_batch(
    node_state, node_legal, node_P, node_N, node_W, node_child, node_terminal,
    node_count, path_node, path_action, path_len, leaf_state, leaf_node,
    needs_eval, leaf_value, scratch_all, c_puct, fpu_reduction, max_nodes,
):
    """Descend every tree to a leaf, expanding one node per game."""
    G = node_state.shape[0]
    for g in prange(G):
        scratch = scratch_all[g]
        node = 0
        depth = 0
        needs_eval[g] = 0
        leaf_value[g] = 0.0
        leaf_node[g] = -1

        while True:
            if node_terminal[g, node] >= 0:
                # Whoever is to move here has already been beaten to the goal.
                leaf_value[g] = -1.0
                leaf_node[g] = node
                break

            total = 0.0
            wsum = 0.0
            visited_prior = 0.0
            for a in range(NUM_ACTIONS):
                na = node_N[g, node, a]
                total += na
                if na > 0.0:
                    wsum += node_W[g, node, a]
                    visited_prior += node_P[g, node, a]

            # First-play urgency. Quoridor has up to 140 legal actions, so an
            # optimistic default for unvisited edges makes PUCT sweep the entire
            # action space before revisiting anything -- at realistic simulation
            # counts the root visit distribution would come out nearly uniform
            # and carry no training signal. Assuming an unvisited move is
            # somewhat worse than the node's current estimate makes search
            # concentrate on the moves that already look good.
            parent_q = wsum / total if total > 0.0 else 0.0
            fpu_val = parent_q - fpu_reduction * np.sqrt(visited_prior)

            # max(total, 1) so the first visit to a node still orders by prior
            # instead of collapsing every U term to zero.
            sqrt_total = np.sqrt(total) if total > 1.0 else 1.0

            best_a = -1
            best_s = -1e30
            for a in range(NUM_ACTIONS):
                if node_legal[g, node, a] == 0:
                    continue
                n = node_N[g, node, a]
                q = node_W[g, node, a] / n if n > 0.0 else fpu_val
                u = c_puct * node_P[g, node, a] * sqrt_total / (1.0 + n)
                s = q + u
                if s > best_s:
                    best_s = s
                    best_a = a
            if best_a < 0:  # no legal action (cannot occur in Quoridor)
                break

            path_node[g, depth] = node
            path_action[g, depth] = best_a
            depth += 1

            child = node_child[g, node, best_a]
            if child < 0:
                if node_count[g] >= max_nodes:
                    break  # arena exhausted; back up a neutral value
                new = node_count[g]
                node_count[g] += 1
                for i in range(STATE_SIZE):
                    node_state[g, new, i] = node_state[g, node, i]
                apply_action(node_state[g, new], best_a, scratch)
                for a in range(NUM_ACTIONS):
                    node_N[g, new, a] = 0.0
                    node_W[g, new, a] = 0.0
                    node_P[g, new, a] = 0.0
                    node_child[g, new, a] = -1
                    node_legal[g, new, a] = 0
                node_child[g, node, best_a] = new

                w = winner(node_state[g, new])
                node_terminal[g, new] = w
                leaf_node[g] = new
                if w >= 0:
                    leaf_value[g] = -1.0
                else:
                    legal_mask(node_state[g, new], node_legal[g, new], scratch)
                    needs_eval[g] = 1
                    for i in range(STATE_SIZE):
                        leaf_state[g, i] = node_state[g, new, i]
                break

            node = child
            if depth >= MAX_DEPTH:
                break
        path_len[g] = depth


@njit(cache=True, parallel=True)
def _backup_batch(
    node_P, node_N, node_W, path_node, path_action, path_len,
    leaf_node, needs_eval, leaf_value, policy, value,
):
    """Install priors on freshly expanded leaves and propagate values to the root."""
    G = path_node.shape[0]
    for g in prange(G):
        v = leaf_value[g]
        if needs_eval[g] != 0:
            ln = leaf_node[g]
            for a in range(NUM_ACTIONS):
                node_P[g, ln, a] = policy[g, a]
            v = value[g]
        for i in range(path_len[g] - 1, -1, -1):
            v = -v  # flip to the perspective of the mover one ply up
            n = path_node[g, i]
            a = path_action[g, i]
            node_N[g, n, a] += 1.0
            node_W[g, n, a] += v


@njit(cache=True, parallel=True)
def _encode_batch(states, out, legal_in, legal_canon, flipped, scratch_all,
                  d0_all, d1_all, canon_all, rot):
    """Canonicalise, encode, and rotate legal masks into canonical action space."""
    G = states.shape[0]
    for g in prange(G):
        was_flipped = canonicalize(states[g], canon_all[g])
        flipped[g] = 1 if was_flipped else 0
        encode_canonical(canon_all[g], out[g], scratch_all[g], d0_all[g], d1_all[g])
        if was_flipped:
            for a in range(NUM_ACTIONS):
                legal_canon[g, a] = legal_in[g, rot[a]]
        else:
            for a in range(NUM_ACTIONS):
                legal_canon[g, a] = legal_in[g, a]


@njit(cache=True, parallel=True)
def _init_roots(states, node_state, node_legal, node_N, node_W, node_P,
                node_child, node_terminal, node_count, scratch_all):
    G = states.shape[0]
    for g in prange(G):
        for i in range(STATE_SIZE):
            node_state[g, 0, i] = states[g, i]
        for a in range(NUM_ACTIONS):
            node_N[g, 0, a] = 0.0
            node_W[g, 0, a] = 0.0
            node_P[g, 0, a] = 0.0
            node_child[g, 0, a] = -1
            node_legal[g, 0, a] = 0
        node_count[g] = 1
        w = winner(node_state[g, 0])
        node_terminal[g, 0] = w
        if w < 0:
            legal_mask(node_state[g, 0], node_legal[g, 0], scratch_all[g])


class BatchedMCTS:
    """PUCT search over ``n_games`` independent trees, sharing one GPU batch."""

    def __init__(
        self,
        evaluator,
        n_games: int = 128,
        max_nodes: int = 800,
        c_puct: float = 1.6,
        # Tuned for a *trained*, peaked prior. Under a uniform prior every U term
        # is ~1/n_actions and the FPU penalty swamps it, collapsing search onto
        # whichever action happens to be tried first -- so bootstrap the policy
        # before relying on this, or set it to 0.
        fpu_reduction: float = 0.2,
        dirichlet_alpha: float = 0.15,
        dirichlet_frac: float = 0.25,
        seed: int = 0,
    ):
        self.evaluator = evaluator
        self.G = n_games
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.rng = np.random.default_rng(seed)

        G, M, A = n_games, max_nodes, NUM_ACTIONS
        self.node_state = np.zeros((G, M, STATE_SIZE), dtype=np.uint8)
        self.node_legal = np.zeros((G, M, A), dtype=np.uint8)
        self.node_P = np.zeros((G, M, A), dtype=np.float32)
        self.node_N = np.zeros((G, M, A), dtype=np.float32)
        self.node_W = np.zeros((G, M, A), dtype=np.float32)
        self.node_child = np.full((G, M, A), -1, dtype=np.int32)
        self.node_terminal = np.full((G, M), -1, dtype=np.int8)
        self.node_count = np.zeros(G, dtype=np.int32)

        self.path_node = np.zeros((G, MAX_DEPTH), dtype=np.int32)
        self.path_action = np.zeros((G, MAX_DEPTH), dtype=np.int32)
        self.path_len = np.zeros(G, dtype=np.int32)

        self.leaf_state = np.zeros((G, STATE_SIZE), dtype=np.uint8)
        self.leaf_node = np.zeros(G, dtype=np.int32)
        self.needs_eval = np.zeros(G, dtype=np.uint8)
        self.leaf_value = np.zeros(G, dtype=np.float32)

        self.scratch_all = np.zeros((G, SCRATCH_SIZE), dtype=np.int32)
        self.d0_all = np.zeros((G, 81), dtype=np.int32)
        self.d1_all = np.zeros((G, 81), dtype=np.int32)
        self.canon_all = np.zeros((G, STATE_SIZE), dtype=np.uint8)
        self.planes = np.zeros((G, NUM_PLANES, 9, 9), dtype=np.float32)
        self.legal_canon = np.zeros((G, A), dtype=np.uint8)
        self.flipped = np.zeros(G, dtype=np.uint8)
        self._rot = ROT180_ACTION.astype(np.int32)

    # ------------------------------------------------------------------ core

    def _evaluate(self, states: np.ndarray, legal: np.ndarray):
        """Encode, run the network, and map policies back to real action space."""
        n = states.shape[0]
        _encode_batch(
            states, self.planes[:n], legal, self.legal_canon[:n], self.flipped[:n],
            self.scratch_all[:n], self.d0_all[:n], self.d1_all[:n],
            self.canon_all[:n], self._rot,
        )
        policy, value = self.evaluator.evaluate(self.planes[:n], self.legal_canon[:n])
        flip_idx = np.nonzero(self.flipped[:n])[0]
        if len(flip_idx):
            policy[flip_idx] = policy[flip_idx][:, self._rot]
        return policy, value

    def _init_roots(self, states: np.ndarray, add_noise: bool, n: int) -> None:
        _init_roots(
            states, self.node_state[:n], self.node_legal[:n], self.node_N[:n],
            self.node_W[:n], self.node_P[:n], self.node_child[:n],
            self.node_terminal[:n], self.node_count[:n], self.scratch_all[:n],
        )
        live = np.nonzero(self.node_terminal[:n, 0] < 0)[0]
        if len(live) == 0:
            return
        policy, _ = self._evaluate(
            self.node_state[live, 0], self.node_legal[live, 0]
        )
        if add_noise:
            policy = self._apply_dirichlet(policy, self.node_legal[live, 0])
        self.node_P[live, 0] = policy

    def _apply_dirichlet(self, policy: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """Mix Dirichlet noise into root priors, over legal actions only."""
        out = policy.copy()
        for i in range(policy.shape[0]):
            idx = np.nonzero(legal[i])[0]
            if len(idx) == 0:
                continue
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(idx))
            out[i, idx] = (
                (1 - self.dirichlet_frac) * policy[i, idx]
                + self.dirichlet_frac * noise
            )
        return out

    def search(
        self, states: np.ndarray, n_sims: int, add_noise: bool = False
    ) -> np.ndarray:
        """Run ``n_sims`` simulations per game; return root visit counts [n,140].

        Accepts any batch up to the configured capacity, so callers can shrink the
        batch as games finish instead of searching dead boards.
        """
        n = states.shape[0]
        assert n <= self.G, f"capacity is {self.G} games, got {n}"
        self._init_roots(states, add_noise, n)
        self._last_n = n

        for _ in range(n_sims):
            _select_batch(
                self.node_state[:n], self.node_legal[:n], self.node_P[:n],
                self.node_N[:n], self.node_W[:n], self.node_child[:n],
                self.node_terminal[:n], self.node_count[:n], self.path_node[:n],
                self.path_action[:n], self.path_len[:n], self.leaf_state[:n],
                self.leaf_node[:n], self.needs_eval[:n], self.leaf_value[:n],
                self.scratch_all[:n], self.c_puct, self.fpu_reduction, self.max_nodes,
            )
            idx = np.nonzero(self.needs_eval[:n])[0]
            policy = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
            value = np.zeros(n, dtype=np.float32)
            if len(idx):
                leaf_legal = self.node_legal[idx, self.leaf_node[idx]]
                p, v = self._evaluate(self.leaf_state[idx], leaf_legal)
                policy[idx] = p
                value[idx] = v
            _backup_batch(
                self.node_P[:n], self.node_N[:n], self.node_W[:n], self.path_node[:n],
                self.path_action[:n], self.path_len[:n], self.leaf_node[:n],
                self.needs_eval[:n], self.leaf_value[:n], policy, value,
            )
        return self.node_N[:n, 0, :].copy()

    def root_value(self) -> np.ndarray:
        """Mean value of the root, from the side-to-move's perspective."""
        n = getattr(self, "_last_n", self.G)
        counts = self.node_N[:n, 0, :]
        w = self.node_W[:n, 0, :]
        total = counts.sum(axis=1)
        return np.where(total > 0, w.sum(axis=1) / np.maximum(total, 1), 0.0)


def visits_to_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
    """Normalise visit counts into a move distribution at the given temperature."""
    if temperature <= 1e-3:
        out = np.zeros_like(visits)
        out[np.arange(len(visits)), visits.argmax(axis=1)] = 1.0
        return out
    scaled = np.power(visits, 1.0 / temperature)
    total = scaled.sum(axis=1, keepdims=True)
    return np.divide(scaled, np.maximum(total, 1e-12))


class MCTSAgent:
    """Single-game agent wrapper around :class:`BatchedMCTS`, for arena play."""

    def __init__(
        self,
        evaluator,
        n_sims: int = 400,
        c_puct: float = 1.6,
        temperature: float = 0.0,
        max_nodes: int = 800,
        seed: int = 0,
        name: str | None = None,
    ):
        self.name = name or f"mcts-{n_sims}"
        self.n_sims = n_sims
        self.temperature = temperature
        self.mcts = BatchedMCTS(
            evaluator, n_games=1, max_nodes=max_nodes, c_puct=c_puct, seed=seed
        )
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_action(self, st: np.ndarray) -> int:
        visits = self.mcts.search(st.reshape(1, -1).copy(), self.n_sims)
        if self.temperature <= 1e-3:
            return int(visits[0].argmax())
        probs = visits_to_policy(visits, self.temperature)[0]
        return int(self.rng.choice(len(probs), p=probs))
