"""Numba-jitted Quoridor core: move generation, wall legality, and BFS.

This is the hot path for MCTS. It mirrors :mod:`quoridor.pyrules` exactly and is
validated against it by differential testing in ``tests/test_rules.py``.

State layout
------------
A position is a single ``uint8`` array of length :data:`STATE_SIZE`, so copying a
node during search is one small ``memcpy``::

    [  0: 64]  horizontal walls, 8x8 slot grid, row-major
    [ 64:128]  vertical walls, 8x8 slot grid, row-major
    [128]      player 0 pawn cell (0..80)
    [129]      player 1 pawn cell (0..80)
    [130]      player 0 walls remaining
    [131]      player 1 walls remaining
    [132]      side to move (0 or 1)

Wall legality
-------------
Validating the 128 candidate wall placements naively would need 128 connectivity
searches. Two optimisations cut that to near zero:

1. **Path witness.** Compute one concrete shortest path per player and mark its
   edges. A candidate that blocks no marked edge cannot disconnect that player --
   the witnessed path survives -- so only path-crossing candidates need checking,
   and only for the player whose path they actually cross.

2. **Bitboard flood fill.** The remaining checks use an 81-bit occupancy set held
   in two ``uint64`` words (cells 0-63 in ``lo``, 64-80 in ``hi``). One flood
   iteration expands every frontier cell in all four directions with a handful of
   shifts and masks, instead of walking a queue cell by cell. The per-direction
   passability masks are derived once per position straight from the wall arrays
   and then patched in place for each candidate.

Edge indexing (used by the path witness)
-----------------------------------------
There are 144 board edges::

    vertical  edge (r,c)-(r+1,c) -> index r * 9 + c        for r in 0..7, c in 0..8
    horizontal edge (r,c)-(r,c+1) -> index 72 + r * 8 + c   for r in 0..8, c in 0..7
"""

from __future__ import annotations

import numpy as np
from numba import njit

N = 9
W = 8
NUM_CELLS = 81
NUM_WALL_SLOTS = 64
NUM_ACTIONS = 140
MOVE_BASE = 128
NUM_EDGES = 144
INF = 1_000_000

STATE_SIZE = 133
WH_OFF = 0
WV_OFF = 64
IDX_P0 = 128
IDX_P1 = 129
IDX_WL0 = 130
IDX_WL1 = 131
IDX_TURN = 132

WALLS_PER_PLAYER = 10
START_CELLS = (4, 76)  # (0,4) and (8,4)
GOAL_ROWS = (8, 0)

# Scratch layout (int32); see make_scratch().
SCRATCH_SIZE = 640
_S_DIST0, _S_DIST1, _S_TMP, _S_QUEUE = 0, 81, 162, 243
_S_ONPATH0, _S_ONPATH1, _S_DESTS = 324, 468, 612

DR = np.array([-1, 1, 0, 0], dtype=np.int32)
DC = np.array([0, 0, 1, -1], dtype=np.int32)
# Perpendicular directions for each orthogonal direction (N/S -> E,W; E/W -> N,S).
PERP = np.array([[2, 3], [2, 3], [0, 1], [0, 1]], dtype=np.int32)

# --- bitboard constants -----------------------------------------------------
U1 = np.uint64(1)
U9 = np.uint64(9)
U55 = np.uint64(55)
U63 = np.uint64(63)
MASK_HI = np.uint64((1 << 17) - 1)  # cells 64..80 live in the low 17 bits of `hi`


def _base_masks() -> np.ndarray:
    """Passability masks for an empty board (board edges only), as [lo, hi] pairs.

    Order is N, S, E, W -- matching DR/DC -- giving an ``uint64[8]`` array.
    """
    can = [0, 0, 0, 0]
    for r in range(N):
        for c in range(N):
            i = r * N + c
            if r > 0:
                can[0] |= 1 << i
            if r < N - 1:
                can[1] |= 1 << i
            if c < N - 1:
                can[2] |= 1 << i
            if c > 0:
                can[3] |= 1 << i
    out = np.zeros(8, dtype=np.uint64)
    for d in range(4):
        out[2 * d] = np.uint64(can[d] & 0xFFFFFFFFFFFFFFFF)
        out[2 * d + 1] = np.uint64(can[d] >> 64)
    return out


BASE_MASKS = _base_masks()

# Goal-row occupancy masks, indexed by player (0 -> row 8, 1 -> row 0).
def _goal_masks() -> np.ndarray:
    out = np.zeros((2, 2), dtype=np.uint64)
    for p, gr in enumerate(GOAL_ROWS):
        bits = 0
        for c in range(N):
            bits |= 1 << (gr * N + c)
        out[p, 0] = np.uint64(bits & 0xFFFFFFFFFFFFFFFF)
        out[p, 1] = np.uint64(bits >> 64)
    return out


GOAL_MASKS = _goal_masks()


def make_scratch() -> np.ndarray:
    return np.zeros(SCRATCH_SIZE, dtype=np.int32)


def initial_state() -> np.ndarray:
    st = np.zeros(STATE_SIZE, dtype=np.uint8)
    st[IDX_P0] = START_CELLS[0]
    st[IDX_P1] = START_CELLS[1]
    st[IDX_WL0] = WALLS_PER_PLAYER
    st[IDX_WL1] = WALLS_PER_PLAYER
    st[IDX_TURN] = 0
    return st


# --------------------------------------------------------------------- walls


@njit(cache=True, inline="always")
def _h_at(st, wr, wc):
    if wr < 0 or wr >= W or wc < 0 or wc >= W:
        return False
    return st[WH_OFF + wr * W + wc] != 0


@njit(cache=True, inline="always")
def _v_at(st, wr, wc):
    if wr < 0 or wr >= W or wc < 0 or wc >= W:
        return False
    return st[WV_OFF + wr * W + wc] != 0


@njit(cache=True, inline="always")
def _blocked(st, r, c, d):
    """True if the orthogonal step ``d`` from ``(r, c)`` is off-board or walled."""
    if d == 0:  # N -- edge between rows r-1 and r
        if r - 1 < 0:
            return True
        return _h_at(st, r - 1, c) or _h_at(st, r - 1, c - 1)
    elif d == 1:  # S -- edge between rows r and r+1
        if r + 1 >= N:
            return True
        return _h_at(st, r, c) or _h_at(st, r, c - 1)
    elif d == 2:  # E -- edge between cols c and c+1
        if c + 1 >= N:
            return True
        return _v_at(st, r, c) or _v_at(st, r - 1, c)
    else:  # W -- edge between cols c-1 and c
        if c - 1 < 0:
            return True
        return _v_at(st, r, c - 1) or _v_at(st, r - 1, c - 1)


# ----------------------------------------------------------------- bitboards


@njit(cache=True, inline="always")
def _clear_bit(m, d, cell):
    if cell < 64:
        m[2 * d] &= ~(U1 << np.uint64(cell))
    else:
        m[2 * d + 1] &= ~(U1 << np.uint64(cell - 64))


@njit(cache=True, inline="always")
def _set_bit(m, d, cell):
    if cell < 64:
        m[2 * d] |= U1 << np.uint64(cell)
    else:
        m[2 * d + 1] |= U1 << np.uint64(cell - 64)


@njit(cache=True)
def _build_masks(st, m):
    """Derive the four passability bitboards for the current wall configuration."""
    for i in range(8):
        m[i] = BASE_MASKS[i]
    for wr in range(W):
        for wc in range(W):
            if st[WH_OFF + wr * W + wc] != 0:
                # Blocks S out of the upper two cells, N out of the lower two.
                _clear_bit(m, 1, wr * N + wc)
                _clear_bit(m, 1, wr * N + wc + 1)
                _clear_bit(m, 0, (wr + 1) * N + wc)
                _clear_bit(m, 0, (wr + 1) * N + wc + 1)
            if st[WV_OFF + wr * W + wc] != 0:
                # Blocks E out of the left two cells, W out of the right two.
                _clear_bit(m, 2, wr * N + wc)
                _clear_bit(m, 2, (wr + 1) * N + wc)
                _clear_bit(m, 3, wr * N + wc + 1)
                _clear_bit(m, 3, (wr + 1) * N + wc + 1)


@njit(cache=True, inline="always")
def _patch_wall(m, orient, wr, wc, place):
    """Apply (``place``) or undo the passability edits for one wall slot.

    Safe to undo by re-setting: the caller has already rejected overlapping and
    crossing walls, so no other wall clears these same bits.
    """
    if orient == 0:
        c0, c1 = wr * N + wc, wr * N + wc + 1
        c2, c3 = (wr + 1) * N + wc, (wr + 1) * N + wc + 1
        if place:
            _clear_bit(m, 1, c0)
            _clear_bit(m, 1, c1)
            _clear_bit(m, 0, c2)
            _clear_bit(m, 0, c3)
        else:
            _set_bit(m, 1, c0)
            _set_bit(m, 1, c1)
            _set_bit(m, 0, c2)
            _set_bit(m, 0, c3)
    else:
        c0, c1 = wr * N + wc, (wr + 1) * N + wc
        c2, c3 = wr * N + wc + 1, (wr + 1) * N + wc + 1
        if place:
            _clear_bit(m, 2, c0)
            _clear_bit(m, 2, c1)
            _clear_bit(m, 3, c2)
            _clear_bit(m, 3, c3)
        else:
            _set_bit(m, 2, c0)
            _set_bit(m, 2, c1)
            _set_bit(m, 3, c2)
            _set_bit(m, 3, c3)


@njit(cache=True)
def _flood_reaches(m, player, target_cell):
    """True if ``target_cell`` connects to ``player``'s goal row.

    Floods outward from the whole goal row and stops as soon as the target is
    swallowed, which is typically well before the board is exhausted.
    """
    rlo = GOAL_MASKS[player, 0]
    rhi = GOAL_MASKS[player, 1]
    if target_cell < 64:
        tlo = U1 << np.uint64(target_cell)
        thi = np.uint64(0)
    else:
        tlo = np.uint64(0)
        thi = U1 << np.uint64(target_cell - 64)

    while True:
        if (rlo & tlo) != 0 or (rhi & thi) != 0:
            return True
        # North: (reach & canN) >> 9
        alo = rlo & m[0]
        ahi = rhi & m[1]
        nlo = (alo >> U9) | (ahi << U55)
        nhi = ahi >> U9
        # South: (reach & canS) << 9
        blo = rlo & m[2]
        bhi = rhi & m[3]
        slo = blo << U9
        shi = ((bhi << U9) | (blo >> U55)) & MASK_HI
        # East: (reach & canE) << 1
        clo = rlo & m[4]
        chi = rhi & m[5]
        elo = clo << U1
        ehi = ((chi << U1) | (clo >> U63)) & MASK_HI
        # West: (reach & canW) >> 1
        dlo = rlo & m[6]
        dhi = rhi & m[7]
        wlo = (dlo >> U1) | (dhi << U63)
        whi = dhi >> U1

        newlo = (nlo | slo | elo | wlo) & ~rlo
        newhi = (nhi | shi | ehi | whi) & ~rhi & MASK_HI
        if newlo == 0 and newhi == 0:
            return False
        rlo |= newlo
        rhi |= newhi


# ---------------------------------------------------------------------- BFS


@njit(cache=True)
def _distance_map(st, goal_row, dist, queue):
    """Multi-source BFS: distance from every cell to ``goal_row``."""
    for i in range(NUM_CELLS):
        dist[i] = INF
    head = 0
    tail = 0
    for c in range(N):
        cell = goal_row * N + c
        dist[cell] = 0
        queue[tail] = cell
        tail += 1
    while head < tail:
        cell = queue[head]
        head += 1
        r = cell // N
        c = cell - r * N
        nd = dist[cell] + 1
        for d in range(4):
            if _blocked(st, r, c, d):
                continue
            ncell = (r + DR[d]) * N + (c + DC[d])
            if dist[ncell] != INF:
                continue
            dist[ncell] = nd
            queue[tail] = ncell
            tail += 1


@njit(cache=True)
def _mark_path_edges(st, dist, start, on_path):
    """Mark the edges of one concrete shortest path from ``start`` to the goal."""
    cur = start
    if dist[cur] >= INF:
        return
    while dist[cur] > 0:
        r = cur // N
        c = cur - r * N
        moved = False
        for d in range(4):
            if _blocked(st, r, c, d):
                continue
            nr = r + DR[d]
            nc = c + DC[d]
            ncell = nr * N + nc
            if dist[ncell] != dist[cur] - 1:
                continue
            if d == 0:
                e = nr * N + c  # vertical edge (r-1,c)-(r,c)
            elif d == 1:
                e = r * N + c  # vertical edge (r,c)-(r+1,c)
            elif d == 2:
                e = 72 + r * W + c  # horizontal edge (r,c)-(r,c+1)
            else:
                e = 72 + r * W + nc  # horizontal edge (r,c-1)-(r,c)
            on_path[e] = 1
            cur = ncell
            moved = True
            break
        if not moved:  # unreachable: a finite distance always has a successor
            break


@njit(cache=True)
def distance_map(st, player, dist, scratch):
    queue = scratch[_S_QUEUE:_S_QUEUE + 81]
    _distance_map(st, 8 if player == 0 else 0, dist, queue)


@njit(cache=True)
def path_edge_mask(st, player, on_path, scratch):
    """Mark (in ``on_path``, uint8[144]) the edges of ``player``'s shortest path.

    Used by the heuristic agent to pick wall candidates: the only walls that do
    anything immediately are the ones cutting an edge of the opponent's route.
    """
    dist = scratch[_S_DIST0:_S_DIST0 + 81]
    queue = scratch[_S_QUEUE:_S_QUEUE + 81]
    for i in range(NUM_EDGES):
        on_path[i] = 0
    _distance_map(st, 8 if player == 0 else 0, dist, queue)
    _mark_path_edges(st, dist, st[IDX_P0] if player == 0 else st[IDX_P1], on_path)


@njit(cache=True)
def shortest_path_len(st, player, scratch):
    dist = scratch[_S_DIST0:_S_DIST0 + 81]
    queue = scratch[_S_QUEUE:_S_QUEUE + 81]
    _distance_map(st, 8 if player == 0 else 0, dist, queue)
    return dist[st[IDX_P0] if player == 0 else st[IDX_P1]]


# -------------------------------------------------------------- pawn moves


@njit(cache=True)
def _legal_pawn_moves(st, mask, dests):
    """Fill ``mask[128:140]`` and ``dests[0:12]`` for the side to move."""
    turn = st[IDX_TURN]
    cur = st[IDX_P0] if turn == 0 else st[IDX_P1]
    opp = st[IDX_P1] if turn == 0 else st[IDX_P0]
    r = cur // N
    c = cur - r * N
    for d in range(4):
        if _blocked(st, r, c, d):
            continue
        nr = r + DR[d]
        nc = c + DC[d]
        ncell = nr * N + nc
        if ncell != opp:
            mask[MOVE_BASE + d] = 1
            dests[d] = ncell
            continue
        # Opponent adjacent: prefer the straight jump over them.
        if not _blocked(st, nr, nc, d):
            mask[MOVE_BASE + 4 + d] = 1
            dests[4 + d] = (nr + DR[d]) * N + (nc + DC[d])
        else:
            # Straight jump unavailable -> diagonal side-steps.
            for pi in range(2):
                pd = PERP[d, pi]
                if _blocked(st, nr, nc, pd):
                    continue
                kr = nr + DR[pd]
                kc = nc + DC[pd]
                ddr = DR[d] + DR[pd]
                ddc = DC[d] + DC[pd]
                if ddr == -1 and ddc == 1:
                    off = 8
                elif ddr == -1 and ddc == -1:
                    off = 9
                elif ddr == 1 and ddc == 1:
                    off = 10
                else:
                    off = 11
                mask[MOVE_BASE + off] = 1
                dests[off] = kr * N + kc


# ------------------------------------------------------------- wall moves


@njit(cache=True)
def _legal_wall_moves(st, mask, scratch, masks):
    turn = st[IDX_TURN]
    if (st[IDX_WL0] if turn == 0 else st[IDX_WL1]) <= 0:
        return

    dist0 = scratch[_S_DIST0:_S_DIST0 + 81]
    dist1 = scratch[_S_DIST1:_S_DIST1 + 81]
    queue = scratch[_S_QUEUE:_S_QUEUE + 81]
    on0 = scratch[_S_ONPATH0:_S_ONPATH0 + NUM_EDGES]
    on1 = scratch[_S_ONPATH1:_S_ONPATH1 + NUM_EDGES]

    _distance_map(st, 8, dist0, queue)
    _distance_map(st, 0, dist1, queue)
    for i in range(NUM_EDGES):
        on0[i] = 0
        on1[i] = 0
    _mark_path_edges(st, dist0, st[IDX_P0], on0)
    _mark_path_edges(st, dist1, st[IDX_P1], on1)

    _build_masks(st, masks)
    p0 = st[IDX_P0]
    p1 = st[IDX_P1]

    for orient in range(2):
        for wr in range(W):
            for wc in range(W):
                if orient == 0:
                    if _h_at(st, wr, wc) or _h_at(st, wr, wc - 1) or _h_at(st, wr, wc + 1):
                        continue
                    if _v_at(st, wr, wc):  # crossing
                        continue
                    e1 = wr * N + wc
                    e2 = wr * N + wc + 1
                else:
                    if _v_at(st, wr, wc) or _v_at(st, wr - 1, wc) or _v_at(st, wr + 1, wc):
                        continue
                    if _h_at(st, wr, wc):  # crossing
                        continue
                    e1 = 72 + wr * W + wc
                    e2 = 72 + (wr + 1) * W + wc

                # Only re-verify the player whose witnessed path this wall cuts.
                hits0 = on0[e1] != 0 or on0[e2] != 0
                hits1 = on1[e1] != 0 or on1[e2] != 0
                ok = True
                if hits0 or hits1:
                    _patch_wall(masks, orient, wr, wc, True)
                    if hits0 and not _flood_reaches(masks, 0, p0):
                        ok = False
                    if ok and hits1 and not _flood_reaches(masks, 1, p1):
                        ok = False
                    _patch_wall(masks, orient, wr, wc, False)
                if ok:
                    mask[orient * NUM_WALL_SLOTS + wr * W + wc] = 1


@njit(cache=True)
def legal_mask(st, mask, scratch):
    """Fill ``mask`` (uint8[140]) with the legal actions for the side to move."""
    for i in range(NUM_ACTIONS):
        mask[i] = 0
    dests = scratch[_S_DESTS:_S_DESTS + 12]
    for i in range(12):
        dests[i] = -1
    _legal_pawn_moves(st, mask, dests)
    masks = np.empty(8, dtype=np.uint64)
    _legal_wall_moves(st, mask, scratch, masks)


@njit(cache=True)
def legal_mask_dests(st, mask, scratch, dests):
    """As :func:`legal_mask`, but also returns pawn destinations in ``dests``."""
    for i in range(NUM_ACTIONS):
        mask[i] = 0
    for i in range(12):
        dests[i] = -1
    _legal_pawn_moves(st, mask, dests)
    masks = np.empty(8, dtype=np.uint64)
    _legal_wall_moves(st, mask, scratch, masks)


@njit(cache=True)
def apply_action(st, action, scratch):
    """Apply ``action`` to ``st`` in place. Assumes the action is legal."""
    turn = st[IDX_TURN]
    if action >= MOVE_BASE:
        dests = scratch[_S_DESTS:_S_DESTS + 12]
        for i in range(12):
            dests[i] = -1
        tmp_mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
        _legal_pawn_moves(st, tmp_mask, dests)
        dest = dests[action - MOVE_BASE]
        if turn == 0:
            st[IDX_P0] = dest
        else:
            st[IDX_P1] = dest
    else:
        orient = action // NUM_WALL_SLOTS
        slot = action - orient * NUM_WALL_SLOTS
        st[(WH_OFF if orient == 0 else WV_OFF) + slot] = 1
        if turn == 0:
            st[IDX_WL0] -= 1
        else:
            st[IDX_WL1] -= 1
    st[IDX_TURN] = 1 - turn


@njit(cache=True)
def winner(st):
    """0 or 1 if that player has reached their goal row, else -1."""
    if st[IDX_P0] // N == 8:
        return 0
    if st[IDX_P1] // N == 0:
        return 1
    return -1


@njit(cache=True)
def random_playout(st, scratch, max_plies, seed_state):
    """Play uniformly random legal moves; return the winner (or -1 on cutoff)."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    rng = seed_state
    for _ in range(max_plies):
        w = winner(st)
        if w >= 0:
            return w
        legal_mask(st, mask, scratch)
        n = 0
        for i in range(NUM_ACTIONS):
            n += mask[i]
        if n == 0:
            return -1
        rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        pick = (rng >> 33) % n
        chosen = -1
        seen = 0
        for i in range(NUM_ACTIONS):
            if mask[i]:
                if seen == pick:
                    chosen = i
                    break
                seen += 1
        apply_action(st, chosen, scratch)
    return -1


def warmup() -> None:
    """Force JIT compilation of the hot functions (first call is slow)."""
    st = initial_state()
    scratch = make_scratch()
    mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    dist = np.zeros(81, dtype=np.int32)
    legal_mask(st, mask, scratch)
    legal_mask_dests(st, mask, scratch, np.zeros(12, dtype=np.int32))
    distance_map(st, 0, dist, scratch)
    shortest_path_len(st, 0, scratch)
    apply_action(st.copy(), 129, scratch)
    random_playout(st.copy(), scratch, 4, 12345)
