"""Agent interface and non-neural baselines.

These exist to bootstrap and to measure. ``HeuristicAgent`` in particular is the
reference opponent the first networks are graded against -- a network that
cannot beat it is not yet worth promoting.

All agents take and return actions in *real* board coordinates (not the
canonical side-to-move frame); canonicalisation is an internal detail of the
neural agent.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from quoridor import fastrules as fr

MATE = 10_000.0


def wall_action_edges(action: int) -> tuple[int, int]:
    """The two board edges a wall action blocks (see fastrules edge indexing)."""
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, fr.W)
    if orient == 0:  # horizontal: two vertical edges side by side
        return wr * fr.N + wc, wr * fr.N + wc + 1
    return 72 + wr * fr.W + wc, 72 + (wr + 1) * fr.W + wc


class Agent(Protocol):
    name: str

    def select_action(self, st: np.ndarray) -> int:
        ...

    def reset(self) -> None:
        ...


class RandomAgent:
    """Uniform over legal actions. The floor that everything must beat."""

    def __init__(self, seed: int = 0, name: str = "random"):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.scratch = fr.make_scratch()
        self._mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)

    def reset(self) -> None:
        pass

    def select_action(self, st: np.ndarray) -> int:
        fr.legal_mask(st, self._mask, self.scratch)
        legal = np.nonzero(self._mask)[0]
        return int(self.rng.choice(legal))


class GreedyAgent:
    """Always steps along its own shortest path; never places a wall.

    Weak, but a useful control: it isolates whether an opponent has learned to
    use walls at all, since a pure racer is beaten only by obstruction.
    """

    def __init__(self, seed: int = 0, name: str = "greedy"):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.scratch = fr.make_scratch()
        self._mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)

    def reset(self) -> None:
        pass

    def select_action(self, st: np.ndarray) -> int:
        fr.legal_mask(st, self._mask, self.scratch)
        me = int(st[fr.IDX_TURN])
        best, best_d = None, None
        for a in np.nonzero(self._mask)[0]:
            if a < fr.MOVE_BASE:
                continue
            child = st.copy()
            fr.apply_action(child, int(a), self.scratch)
            if fr.winner(child) == me:
                return int(a)
            d = fr.shortest_path_len(child, me, self.scratch)
            if best_d is None or d < best_d:
                best, best_d = int(a), d
        if best is None:  # no pawn move available; fall back to any legal action
            return int(np.nonzero(self._mask)[0][0])
        return best


class HeuristicAgent:
    """Alpha-beta over shortest-path difference, with pruned wall candidates.

    The full 140-action branching factor is far too wide for a plain search, so
    wall candidates are restricted to those that actually cut an edge of the
    opponent's current shortest path -- the only walls with an immediate effect.
    Pawn moves are always fully considered.
    """

    def __init__(
        self,
        depth: int = 3,
        root_walls: int = 16,
        inner_walls: int = 8,
        wall_value: float = 0.30,
        seed: int = 0,
        name: str | None = None,
    ):
        self.depth = depth
        self.root_walls = root_walls
        self.inner_walls = inner_walls
        self.wall_value = wall_value
        self.name = name or f"heuristic-d{depth}"
        self.rng = np.random.default_rng(seed)
        self.scratch = fr.make_scratch()
        self._masks = [np.zeros(fr.NUM_ACTIONS, dtype=np.uint8) for _ in range(depth + 2)]
        self._on_path = np.zeros(fr.NUM_EDGES, dtype=np.int32)
        self.nodes = 0

    def reset(self) -> None:
        self.nodes = 0

    # ------------------------------------------------------------ evaluation

    def _eval(self, st: np.ndarray) -> float:
        me = int(st[fr.IDX_TURN])
        opp = 1 - me
        d_me = fr.shortest_path_len(st, me, self.scratch)
        d_opp = fr.shortest_path_len(st, opp, self.scratch)
        if d_me >= fr.INF:
            return -MATE
        if d_opp >= fr.INF:
            return MATE
        walls_me = int(st[fr.IDX_WL0 + me])
        walls_opp = int(st[fr.IDX_WL0 + opp])
        return (d_opp - d_me) + self.wall_value * (walls_me - walls_opp)

    # ------------------------------------------------------------ candidates

    def _candidates(self, st: np.ndarray, depth: int, at_root: bool) -> list[int]:
        mask = self._masks[depth]
        fr.legal_mask(st, mask, self.scratch)
        legal = np.nonzero(mask)[0]
        me = int(st[fr.IDX_TURN])
        opp = 1 - me

        moves = [int(a) for a in legal if a >= fr.MOVE_BASE]
        walls = [int(a) for a in legal if a < fr.MOVE_BASE]

        if walls and int(st[fr.IDX_WL0 + me]) > 0:
            # Only walls that cut the opponent's current route can gain tempo.
            fr.path_edge_mask(st, opp, self._on_path, self.scratch)
            on_path = self._on_path
            walls = [a for a in walls if on_path[wall_action_edges(a)[0]]
                     or on_path[wall_action_edges(a)[1]]]
            cap = self.root_walls if at_root else self.inner_walls
            if at_root and len(walls) > cap:
                walls = self._score_walls(st, walls, me, opp)[:cap]
            else:
                walls = walls[:cap]
        else:
            walls = []

        # Order pawn moves by how much they shorten our own path (best first).
        scored = []
        for a in moves:
            child = st.copy()
            fr.apply_action(child, a, self.scratch)
            if fr.winner(child) == me:
                return [a]  # immediate win, search no further
            scored.append((fr.shortest_path_len(child, me, self.scratch), a))
        scored.sort()
        return [a for _, a in scored] + walls

    def _score_walls(self, st: np.ndarray, walls: list[int], me: int, opp: int) -> list[int]:
        """Rank walls by (opponent delay gained) minus (self delay incurred)."""
        base_me = fr.shortest_path_len(st, me, self.scratch)
        base_opp = fr.shortest_path_len(st, opp, self.scratch)
        scored = []
        for a in walls:
            child = st.copy()
            fr.apply_action(child, a, self.scratch)
            d_me = fr.shortest_path_len(child, me, self.scratch)
            d_opp = fr.shortest_path_len(child, opp, self.scratch)
            gain = (d_opp - base_opp) - (d_me - base_me)
            scored.append((-gain, a))
        scored.sort()
        return [a for _, a in scored]

    # ---------------------------------------------------------------- search

    def _negamax(self, st: np.ndarray, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if fr.winner(st) >= 0:
            # The player to move is the one who just got beaten to the goal.
            return -MATE - depth
        if depth == 0:
            return self._eval(st)

        best = -np.inf
        for a in self._candidates(st, depth, at_root=False):
            child = st.copy()
            fr.apply_action(child, a, self.scratch)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        if best == -np.inf:  # no candidate survived pruning
            return self._eval(st)
        return best

    def select_action(self, st: np.ndarray) -> int:
        cands = self._candidates(st, self.depth + 1, at_root=True)
        if len(cands) == 1:
            return cands[0]
        best_a, best_v = cands[0], -np.inf
        alpha = -np.inf
        for a in cands:
            child = st.copy()
            fr.apply_action(child, a, self.scratch)
            v = -self._negamax(child, self.depth - 1, -np.inf, -alpha)
            if v > best_v:
                best_v, best_a = v, a
            if v > alpha:
                alpha = v
        return best_a
