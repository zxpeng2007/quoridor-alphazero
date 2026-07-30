"""Network input encoding, canonicalisation, and board symmetries.

Canonicalisation
----------------
The network only ever sees positions from the perspective of the side to move,
as "player 0, racing from the low rows down to row 8". When it is player 1's
turn the board is rotated 180 degrees and the two players are swapped, which
halves what the network has to learn. Actions chosen in canonical space are
mapped back with :data:`ROT180_ACTION`.

Symmetries
----------
Quoridor has exactly one non-trivial symmetry that preserves the rules: the
left-right mirror. (Transposes and rotations by 90 degrees do not, because the
two players' goal rows are fixed to the top and bottom.) The mirror is used for
training-data augmentation, doubling the effective sample count for free.

Both symmetries act on the 140-dim action space as permutations, precomputed
here as index arrays.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from quoridor.fastrules import (
    IDX_P0,
    IDX_P1,
    IDX_TURN,
    IDX_WL0,
    IDX_WL1,
    INF,
    MOVE_BASE,
    NUM_ACTIONS,
    NUM_WALL_SLOTS,
    STATE_SIZE,
    WH_OFF,
    WV_OFF,
    _distance_map,
)

N = 9
W = 8
NUM_PLANES = 8
MAX_DIST = 40.0  # distance-map normalisation cap

# Plane meanings, for readability elsewhere.
PLANE_SELF_PAWN = 0
PLANE_OPP_PAWN = 1
PLANE_WALLS_H = 2
PLANE_WALLS_V = 3
PLANE_SELF_WALLS_LEFT = 4
PLANE_OPP_WALLS_LEFT = 5
PLANE_SELF_DIST = 6
PLANE_OPP_DIST = 7


def _build_action_permutations() -> tuple[np.ndarray, np.ndarray]:
    """Action-index permutations for the 180-degree rotation and the mirror."""
    # Pawn-move direction remaps, as offsets into the 12 move actions.
    #   deltas: 0 N, 1 S, 2 E, 3 W, 4 NN, 5 SS, 6 EE, 7 WW, 8 NE, 9 NW, 10 SE, 11 SW
    rot_dir = [1, 0, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8]
    mir_dir = [0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10]

    rot = np.zeros(NUM_ACTIONS, dtype=np.int32)
    mir = np.zeros(NUM_ACTIONS, dtype=np.int32)
    for orient in (0, 1):
        for wr in range(W):
            for wc in range(W):
                src = orient * NUM_WALL_SLOTS + wr * W + wc
                rot[src] = orient * NUM_WALL_SLOTS + (W - 1 - wr) * W + (W - 1 - wc)
                mir[src] = orient * NUM_WALL_SLOTS + wr * W + (W - 1 - wc)
    for d in range(12):
        rot[MOVE_BASE + d] = MOVE_BASE + rot_dir[d]
        mir[MOVE_BASE + d] = MOVE_BASE + mir_dir[d]
    return rot, mir


ROT180_ACTION, MIRROR_ACTION = _build_action_permutations()


# ------------------------------------------------------------ state symmetry


@njit(cache=True)
def rot180_state(st, out):
    """Rotate the board 180 degrees *and* swap the two players' identities.

    Under the rotation a cell ``i`` maps to ``80 - i`` and a wall slot ``(wr,wc)``
    maps to ``(7-wr, 7-wc)``. Player 1, who was racing towards row 0, becomes a
    player racing towards row 8 -- i.e. a player 0.
    """
    for i in range(STATE_SIZE):
        out[i] = 0
    for wr in range(W):
        for wc in range(W):
            src = wr * W + wc
            dst = (W - 1 - wr) * W + (W - 1 - wc)
            out[WH_OFF + dst] = st[WH_OFF + src]
            out[WV_OFF + dst] = st[WV_OFF + src]
    out[IDX_P0] = 80 - st[IDX_P1]
    out[IDX_P1] = 80 - st[IDX_P0]
    out[IDX_WL0] = st[IDX_WL1]
    out[IDX_WL1] = st[IDX_WL0]
    out[IDX_TURN] = 1 - st[IDX_TURN]


@njit(cache=True)
def mirror_state(st, out):
    """Mirror the board left-right. Player identities are unchanged."""
    for i in range(STATE_SIZE):
        out[i] = st[i]
    for wr in range(W):
        for wc in range(W):
            src = wr * W + wc
            dst = wr * W + (W - 1 - wc)
            out[WH_OFF + dst] = st[WH_OFF + src]
            out[WV_OFF + dst] = st[WV_OFF + src]
    p0 = st[IDX_P0]
    p1 = st[IDX_P1]
    out[IDX_P0] = (p0 // N) * N + (N - 1 - p0 % N)
    out[IDX_P1] = (p1 // N) * N + (N - 1 - p1 % N)


@njit(cache=True)
def canonicalize(st, out):
    """Write the side-to-move view of ``st`` into ``out``; True if it was rotated."""
    if st[IDX_TURN] == 0:
        for i in range(STATE_SIZE):
            out[i] = st[i]
        return False
    rot180_state(st, out)
    return True


def to_canonical(st: np.ndarray) -> tuple[np.ndarray, bool]:
    out = np.empty(STATE_SIZE, dtype=np.uint8)
    flipped = canonicalize(st, out)
    return out, flipped


def decanonicalize_action(action: int, flipped: bool) -> int:
    """Map an action chosen in canonical space back to the real board."""
    return int(ROT180_ACTION[action]) if flipped else int(action)


# ----------------------------------------------------------------- encoding


@njit(cache=True)
def encode_canonical(st, out, scratch, d0, d1):
    """Fill ``out`` (float32[8,9,9]) from a canonical state (side to move == 0)."""
    for p in range(NUM_PLANES):
        for r in range(N):
            for c in range(N):
                out[p, r, c] = 0.0

    p0 = st[IDX_P0]
    p1 = st[IDX_P1]
    out[PLANE_SELF_PAWN, p0 // N, p0 % N] = 1.0
    out[PLANE_OPP_PAWN, p1 // N, p1 % N] = 1.0

    for wr in range(W):
        for wc in range(W):
            if st[WH_OFF + wr * W + wc] != 0:
                out[PLANE_WALLS_H, wr, wc] = 1.0
            if st[WV_OFF + wr * W + wc] != 0:
                out[PLANE_WALLS_V, wr, wc] = 1.0

    w0 = st[IDX_WL0] / 10.0
    w1 = st[IDX_WL1] / 10.0
    for r in range(N):
        for c in range(N):
            out[PLANE_SELF_WALLS_LEFT, r, c] = w0
            out[PLANE_OPP_WALLS_LEFT, r, c] = w1

    queue = scratch[243:243 + 81]
    _distance_map(st, 8, d0, queue)  # side to move races to row 8
    _distance_map(st, 0, d1, queue)  # opponent races to row 0
    for i in range(81):
        r = i // N
        c = i % N
        v0 = d0[i]
        v1 = d1[i]
        out[PLANE_SELF_DIST, r, c] = 1.0 if v0 >= INF else min(v0, 40) / MAX_DIST
        out[PLANE_OPP_DIST, r, c] = 1.0 if v1 >= INF else min(v1, 40) / MAX_DIST


def encode(st: np.ndarray) -> np.ndarray:
    """Encode a (possibly non-canonical) state as float32[8,9,9], side-to-move view."""
    from quoridor.fastrules import make_scratch

    canon, _ = to_canonical(st)
    out = np.zeros((NUM_PLANES, N, N), dtype=np.float32)
    encode_canonical(canon, out, make_scratch(),
                     np.zeros(81, dtype=np.int32), np.zeros(81, dtype=np.int32))
    return out


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Mirror encoded planes left-right, matching :data:`MIRROR_ACTION`.

    The wall planes occupy an 8x8 sub-grid of the 9x9 plane, so they mirror about
    column 7 rather than column 8; the cell planes mirror about column 8.
    """
    out = planes[..., ::-1].copy()
    # Undo the 9-wide flip for the wall planes and redo it 8-wide.
    for p in (PLANE_WALLS_H, PLANE_WALLS_V):
        out[..., p, :, :] = 0.0
        out[..., p, :, :W] = planes[..., p, :, :W][..., ::-1]
    return out
