"""Pure-Python reference implementation of Quoridor (barricade.gg) rules.

This module is the *correctness ground truth*. It is written for clarity rather
than speed; the Numba core in ``fastrules.py`` is validated against it by
differential testing in ``tests/test_rules.py``.

Board conventions
-----------------
Cells are a 9x9 grid, ``(row, col)`` with ``row``/``col`` in ``0..8``, flattened
as ``cell = row * 9 + col``.

* Player 0 starts at ``(0, 4)`` and wins by reaching row 8.
* Player 1 starts at ``(8, 4)`` and wins by reaching row 0.
* Each player has 10 walls.

Walls live on an 8x8 grid of *slots*, ``(wr, wc)`` in ``0..7``:

* A **horizontal** wall at slot ``(wr, wc)`` sits between rows ``wr`` and
  ``wr+1``, spanning columns ``wc`` and ``wc+1``. It blocks the two vertical
  edges ``(wr,wc)-(wr+1,wc)`` and ``(wr,wc+1)-(wr+1,wc+1)``.
* A **vertical** wall at slot ``(wr, wc)`` sits between columns ``wc`` and
  ``wc+1``, spanning rows ``wr`` and ``wr+1``. It blocks the two horizontal
  edges ``(wr,wc)-(wr,wc+1)`` and ``(wr+1,wc)-(wr+1,wc+1)``.

Action space (140 discrete actions)
-----------------------------------
* ``0..63``    -- horizontal wall at slot ``(wr, wc)``, index ``wr * 8 + wc``
* ``64..127``  -- vertical wall at slot ``(wr, wc)``, index ``64 + wr * 8 + wc``
* ``128..139`` -- pawn moves, one per entry in :data:`MOVE_DELTAS`
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterator

N = 9  # board is N x N cells
W = N - 1  # wall slots are W x W
NUM_CELLS = N * N  # 81
NUM_WALL_SLOTS = W * W  # 64
NUM_WALL_ACTIONS = 2 * NUM_WALL_SLOTS  # 128
NUM_MOVE_ACTIONS = 12
NUM_ACTIONS = NUM_WALL_ACTIONS + NUM_MOVE_ACTIONS  # 140
MOVE_ACTION_BASE = NUM_WALL_ACTIONS  # 128
WALLS_PER_PLAYER = 10

HORIZONTAL = 0
VERTICAL = 1

INF = 10**6

# The 12 pawn-move deltas, in action order. Indices 0-3 are the orthogonal
# steps, 4-7 the straight jumps over an adjacent opponent, 8-11 the diagonal
# side-steps used when a straight jump is unavailable.
MOVE_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # 0  N
    (1, 0),    # 1  S
    (0, 1),    # 2  E
    (0, -1),   # 3  W
    (-2, 0),   # 4  NN  (jump)
    (2, 0),    # 5  SS  (jump)
    (0, 2),    # 6  EE  (jump)
    (0, -2),   # 7  WW  (jump)
    (-1, 1),   # 8  NE  (diagonal)
    (-1, -1),  # 9  NW  (diagonal)
    (1, 1),    # 10 SE  (diagonal)
    (1, -1),   # 11 SW  (diagonal)
)

# Orthogonal direction indices, and the two perpendicular directions for each.
ORTHO = (0, 1, 2, 3)
PERPENDICULAR: dict[int, tuple[int, int]] = {
    0: (2, 3),  # N -> E, W
    1: (2, 3),  # S -> E, W
    2: (0, 1),  # E -> N, S
    3: (0, 1),  # W -> N, S
}

# Lookup from a combined (dr, dc) delta back to its action offset.
DELTA_TO_MOVE: dict[tuple[int, int], int] = {d: i for i, d in enumerate(MOVE_DELTAS)}

START_CELLS = (0 * N + 4, 8 * N + 4)
GOAL_ROWS = (8, 0)


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < N and 0 <= c < N


def cell_of(r: int, c: int) -> int:
    return r * N + c


def rc_of(cell: int) -> tuple[int, int]:
    return divmod(cell, N)


def slot_index(wr: int, wc: int) -> int:
    return wr * W + wc


def wall_action(orientation: int, wr: int, wc: int) -> int:
    return orientation * NUM_WALL_SLOTS + slot_index(wr, wc)


def decode_action(action: int) -> tuple[str, tuple[int, ...]]:
    """Decode an action index into ``(kind, params)`` for display/debugging."""
    if action >= MOVE_ACTION_BASE:
        return "move", (action - MOVE_ACTION_BASE,)
    orientation, slot = divmod(action, NUM_WALL_SLOTS)
    wr, wc = divmod(slot, W)
    return ("wall_h" if orientation == HORIZONTAL else "wall_v"), (wr, wc)


@dataclass
class State:
    """A complete Quoridor position.

    ``walls_h`` / ``walls_v`` are length-64 lists of 0/1 over the 8x8 slot grid.
    """

    pawns: list[int] = field(default_factory=lambda: list(START_CELLS))
    walls_h: list[int] = field(default_factory=lambda: [0] * NUM_WALL_SLOTS)
    walls_v: list[int] = field(default_factory=lambda: [0] * NUM_WALL_SLOTS)
    walls_left: list[int] = field(default_factory=lambda: [WALLS_PER_PLAYER] * 2)
    turn: int = 0
    move_count: int = 0

    def copy(self) -> "State":
        return replace(
            self,
            pawns=list(self.pawns),
            walls_h=list(self.walls_h),
            walls_v=list(self.walls_v),
            walls_left=list(self.walls_left),
        )

    # ---------------------------------------------------------------- edges

    def blocked(self, r: int, c: int, direction: int) -> bool:
        """True if the orthogonal step ``direction`` from ``(r, c)`` is walled off.

        Assumes the destination is in bounds; callers check that separately.
        """
        if direction == 0:  # N: edge between rows r-1 and r
            wr = r - 1
            if wr < 0:
                return True
            return self._h_at(wr, c) or self._h_at(wr, c - 1)
        if direction == 1:  # S: edge between rows r and r+1
            return self._h_at(r, c) or self._h_at(r, c - 1)
        if direction == 2:  # E: edge between cols c and c+1
            return self._v_at(r, c) or self._v_at(r - 1, c)
        if direction == 3:  # W: edge between cols c-1 and c
            wc = c - 1
            if wc < 0:
                return True
            return self._v_at(r, wc) or self._v_at(r - 1, wc)
        raise ValueError(f"bad direction {direction}")

    def _h_at(self, wr: int, wc: int) -> bool:
        if 0 <= wr < W and 0 <= wc < W:
            return bool(self.walls_h[slot_index(wr, wc)])
        return False

    def _v_at(self, wr: int, wc: int) -> bool:
        if 0 <= wr < W and 0 <= wc < W:
            return bool(self.walls_v[slot_index(wr, wc)])
        return False

    # ----------------------------------------------------------- pawn moves

    def legal_pawn_moves(self, player: int | None = None) -> dict[int, int]:
        """Map ``action index -> destination cell`` for the player's pawn moves."""
        player = self.turn if player is None else player
        cur = self.pawns[player]
        opp = self.pawns[1 - player]
        r, c = rc_of(cur)
        out: dict[int, int] = {}

        for d in ORTHO:
            dr, dc = MOVE_DELTAS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or self.blocked(r, c, d):
                continue
            n_cell = cell_of(nr, nc)
            if n_cell != opp:
                out[MOVE_ACTION_BASE + d] = n_cell
                continue

            # Opponent is adjacent: try the straight jump over them first.
            jr, jc = nr + dr, nc + dc
            if in_bounds(jr, jc) and not self.blocked(nr, nc, d):
                out[MOVE_ACTION_BASE + 4 + d] = cell_of(jr, jc)
            else:
                # Straight jump unavailable -> diagonal side-steps.
                for pd in PERPENDICULAR[d]:
                    pdr, pdc = MOVE_DELTAS[pd]
                    kr, kc = nr + pdr, nc + pdc
                    if not in_bounds(kr, kc) or self.blocked(nr, nc, pd):
                        continue
                    off = DELTA_TO_MOVE[(dr + pdr, dc + pdc)]
                    out[MOVE_ACTION_BASE + off] = cell_of(kr, kc)
        return out

    # --------------------------------------------------------------- search

    def distance_map(self, player: int) -> list[int]:
        """Multi-source BFS distance from every cell to ``player``'s goal row."""
        dist = [INF] * NUM_CELLS
        q: deque[int] = deque()
        goal_row = GOAL_ROWS[player]
        for c in range(N):
            cell = cell_of(goal_row, c)
            dist[cell] = 0
            q.append(cell)
        while q:
            cell = q.popleft()
            r, c = rc_of(cell)
            d0 = dist[cell]
            for d in ORTHO:
                dr, dc = MOVE_DELTAS[d]
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                n_cell = cell_of(nr, nc)
                if dist[n_cell] <= d0 + 1 or self.blocked(r, c, d):
                    continue
                dist[n_cell] = d0 + 1
                q.append(n_cell)
        return dist

    def shortest_path_len(self, player: int) -> int:
        return self.distance_map(player)[self.pawns[player]]

    def has_path(self, player: int) -> bool:
        return self.shortest_path_len(player) < INF

    # ---------------------------------------------------------------- walls

    def wall_conflicts(self, orientation: int, wr: int, wc: int) -> bool:
        """True if the wall overlaps or crosses an existing wall (ignores paths)."""
        if orientation == HORIZONTAL:
            if self._h_at(wr, wc) or self._h_at(wr, wc - 1) or self._h_at(wr, wc + 1):
                return True
            return self._v_at(wr, wc)
        if self._v_at(wr, wc) or self._v_at(wr - 1, wc) or self._v_at(wr + 1, wc):
            return True
        return self._h_at(wr, wc)

    def _place_wall_unchecked(self, orientation: int, wr: int, wc: int) -> None:
        target = self.walls_h if orientation == HORIZONTAL else self.walls_v
        target[slot_index(wr, wc)] = 1

    def _remove_wall(self, orientation: int, wr: int, wc: int) -> None:
        target = self.walls_h if orientation == HORIZONTAL else self.walls_v
        target[slot_index(wr, wc)] = 0

    def legal_wall_actions(self, player: int | None = None) -> list[int]:
        """All legal wall placements for ``player``.

        Uses the standard *path-witness* shortcut: a candidate wall that blocks
        no edge of an already-known shortest path cannot disconnect that player,
        so the expensive connectivity BFS only runs for the handful of walls
        that actually intersect a current path.
        """
        player = self.turn if player is None else player
        if self.walls_left[player] <= 0:
            return []

        witness = [self._path_edges(0), self._path_edges(1)]
        out: list[int] = []
        for orientation in (HORIZONTAL, VERTICAL):
            for wr in range(W):
                for wc in range(W):
                    if self.wall_conflicts(orientation, wr, wc):
                        continue
                    edges = _wall_edges(orientation, wr, wc)
                    touches = any(edges & witness[p] for p in (0, 1))
                    if touches:
                        self._place_wall_unchecked(orientation, wr, wc)
                        ok = self.has_path(0) and self.has_path(1)
                        self._remove_wall(orientation, wr, wc)
                        if not ok:
                            continue
                    out.append(wall_action(orientation, wr, wc))
        return out

    def _path_edges(self, player: int) -> frozenset[tuple[int, int]]:
        """Edge set of one concrete shortest path for ``player`` (empty if none)."""
        dist = self.distance_map(player)
        cur = self.pawns[player]
        if dist[cur] >= INF:
            return frozenset()
        edges: set[tuple[int, int]] = set()
        while dist[cur] > 0:
            r, c = rc_of(cur)
            for d in ORTHO:
                dr, dc = MOVE_DELTAS[d]
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or self.blocked(r, c, d):
                    continue
                n_cell = cell_of(nr, nc)
                if dist[n_cell] == dist[cur] - 1:
                    edges.add((min(cur, n_cell), max(cur, n_cell)))
                    cur = n_cell
                    break
            else:  # pragma: no cover - a finite distance always has a successor
                break
        return frozenset(edges)

    # --------------------------------------------------------------- API

    def legal_actions(self) -> list[int]:
        return sorted(self.legal_pawn_moves()) + self.legal_wall_actions()

    def legal_mask(self) -> list[bool]:
        mask = [False] * NUM_ACTIONS
        for a in self.legal_actions():
            mask[a] = True
        return mask

    def apply(self, action: int) -> "State":
        """Return a new state with ``action`` applied. Does not validate legality."""
        s = self.copy()
        if action >= MOVE_ACTION_BASE:
            dest = self.legal_pawn_moves()[action]
            s.pawns[s.turn] = dest
        else:
            orientation, slot = divmod(action, NUM_WALL_SLOTS)
            wr, wc = divmod(slot, W)
            s._place_wall_unchecked(orientation, wr, wc)
            s.walls_left[s.turn] -= 1
        s.turn = 1 - s.turn
        s.move_count += 1
        return s

    def winner(self) -> int | None:
        for p in (0, 1):
            if rc_of(self.pawns[p])[0] == GOAL_ROWS[p]:
                return p
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None

    # --------------------------------------------------------------- pretty

    def render(self) -> str:
        """ASCII rendering, player 0 as ``A`` (goal: bottom), player 1 as ``B``."""
        rows: list[str] = []
        rows.append("    " + "   ".join(str(c) for c in range(N)))
        for r in range(N):
            line = [f"{r} "]
            for c in range(N):
                cell = cell_of(r, c)
                ch = "A" if cell == self.pawns[0] else "B" if cell == self.pawns[1] else "."
                line.append(f" {ch} ")
                if c < W:
                    line.append("|" if self._v_at(r, c) or self._v_at(r - 1, c) else " ")
            rows.append("".join(line))
            if r < W:
                line = ["  "]
                for c in range(N):
                    line.append("---" if self._h_at(r, c) or self._h_at(r, c - 1) else "   ")
                    if c < W:
                        line.append(" ")
                rows.append("".join(line))
        rows.append(
            f"  walls: A={self.walls_left[0]} B={self.walls_left[1]}   "
            f"turn={'A' if self.turn == 0 else 'B'}  ply={self.move_count}"
        )
        return "\n".join(rows)


def _wall_edges(orientation: int, wr: int, wc: int) -> frozenset[tuple[int, int]]:
    """The two board edges a wall at this slot blocks, as sorted cell pairs."""
    if orientation == HORIZONTAL:
        pairs = [
            (cell_of(wr, wc), cell_of(wr + 1, wc)),
            (cell_of(wr, wc + 1), cell_of(wr + 1, wc + 1)),
        ]
    else:
        pairs = [
            (cell_of(wr, wc), cell_of(wr, wc + 1)),
            (cell_of(wr + 1, wc), cell_of(wr + 1, wc + 1)),
        ]
    return frozenset((min(a, b), max(a, b)) for a, b in pairs)


def initial_state() -> State:
    return State()


def iter_wall_slots() -> Iterator[tuple[int, int, int]]:
    for orientation in (HORIZONTAL, VERTICAL):
        for wr in range(W):
            for wc in range(W):
                yield orientation, wr, wc
