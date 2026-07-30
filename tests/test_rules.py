"""Differential tests: the Numba core must agree with the Python reference exactly.

The random-playout test is the important one -- it compares the full 140-action
legality mask at every ply of many random games, which exercises jumps, diagonal
side-steps, wall overlap/crossing, and the no-full-block constraint far more
thoroughly than hand-written cases can.
"""

from __future__ import annotations

import numpy as np
import pytest

from quoridor import fastrules as fr
from quoridor import pyrules as pr


def to_fast(s: pr.State) -> np.ndarray:
    st = np.zeros(fr.STATE_SIZE, dtype=np.uint8)
    st[fr.WH_OFF:fr.WH_OFF + 64] = np.asarray(s.walls_h, dtype=np.uint8)
    st[fr.WV_OFF:fr.WV_OFF + 64] = np.asarray(s.walls_v, dtype=np.uint8)
    st[fr.IDX_P0] = s.pawns[0]
    st[fr.IDX_P1] = s.pawns[1]
    st[fr.IDX_WL0] = s.walls_left[0]
    st[fr.IDX_WL1] = s.walls_left[1]
    st[fr.IDX_TURN] = s.turn
    return st


def fast_mask(s: pr.State) -> np.ndarray:
    st = to_fast(s)
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(st, mask, fr.make_scratch())
    return mask


def ref_mask(s: pr.State) -> np.ndarray:
    return np.asarray(s.legal_mask(), dtype=np.uint8)


def describe(mask_a: np.ndarray, mask_b: np.ndarray) -> str:
    diff = np.nonzero(mask_a != mask_b)[0]
    return ", ".join(f"{a}={pr.decode_action(int(a))}" for a in diff[:10])


# ------------------------------------------------------------------ opening


def test_initial_position():
    s = pr.initial_state()
    assert s.shortest_path_len(0) == 8
    assert s.shortest_path_len(1) == 8
    # 128 wall placements + S/E/W pawn moves (N is off-board).
    assert len(s.legal_actions()) == 131
    assert sorted(s.legal_pawn_moves()) == [129, 130, 131]


def test_initial_masks_agree():
    s = pr.initial_state()
    assert np.array_equal(ref_mask(s), fast_mask(s))


# -------------------------------------------------------- jumps & diagonals


def _facing(p0_rc, p1_rc, turn=0) -> pr.State:
    s = pr.initial_state()
    s.pawns[0] = pr.cell_of(*p0_rc)
    s.pawns[1] = pr.cell_of(*p1_rc)
    s.turn = turn
    return s


def test_straight_jump_over_opponent():
    s = _facing((4, 4), (5, 4))
    moves = s.legal_pawn_moves()
    assert pr.MOVE_ACTION_BASE + 5 in moves, "SS jump should be available"
    assert moves[pr.MOVE_ACTION_BASE + 5] == pr.cell_of(6, 4)
    # The plain S step onto the occupied square must NOT be offered.
    assert pr.MOVE_ACTION_BASE + 1 not in moves
    assert np.array_equal(ref_mask(s), fast_mask(s))


def test_diagonal_when_wall_behind_opponent():
    s = _facing((4, 4), (5, 4))
    s.walls_h[pr.slot_index(5, 4)] = 1  # blocks (5,4)-(6,4)
    moves = s.legal_pawn_moves()
    assert pr.MOVE_ACTION_BASE + 5 not in moves, "jump is walled off"
    assert moves[pr.MOVE_ACTION_BASE + 10] == pr.cell_of(5, 5)  # SE
    assert moves[pr.MOVE_ACTION_BASE + 11] == pr.cell_of(5, 3)  # SW
    assert np.array_equal(ref_mask(s), fast_mask(s))


def test_diagonal_when_opponent_on_board_edge():
    s = _facing((7, 4), (8, 4))
    moves = s.legal_pawn_moves()
    assert pr.MOVE_ACTION_BASE + 5 not in moves, "jump would leave the board"
    assert moves[pr.MOVE_ACTION_BASE + 10] == pr.cell_of(8, 5)  # SE
    assert moves[pr.MOVE_ACTION_BASE + 11] == pr.cell_of(8, 3)  # SW
    assert np.array_equal(ref_mask(s), fast_mask(s))


def test_diagonal_suppressed_by_side_wall():
    s = _facing((7, 4), (8, 4))
    s.walls_v[pr.slot_index(7, 4)] = 1  # blocks (8,4)-(8,5), killing the SE step
    moves = s.legal_pawn_moves()
    assert pr.MOVE_ACTION_BASE + 10 not in moves
    assert moves[pr.MOVE_ACTION_BASE + 11] == pr.cell_of(8, 3)
    assert np.array_equal(ref_mask(s), fast_mask(s))


# ------------------------------------------------------------------- walls


def test_wall_overlap_and_crossing():
    s = pr.initial_state()
    s.walls_h[pr.slot_index(3, 3)] = 1
    assert s.wall_conflicts(pr.HORIZONTAL, 3, 3), "same slot"
    assert s.wall_conflicts(pr.HORIZONTAL, 3, 2), "overlapping to the left"
    assert s.wall_conflicts(pr.HORIZONTAL, 3, 4), "overlapping to the right"
    assert s.wall_conflicts(pr.VERTICAL, 3, 3), "crossing"
    assert not s.wall_conflicts(pr.HORIZONTAL, 3, 1)
    assert not s.wall_conflicts(pr.VERTICAL, 3, 4)
    assert np.array_equal(ref_mask(s), fast_mask(s))


def test_wall_may_not_fully_block_a_player():
    """Seal player 0 into the top-left corner region except for one gap."""
    s = pr.initial_state()
    s.pawns[0] = pr.cell_of(0, 0)
    # Wall off the row-0/row-1 boundary across columns 0..5.
    for wc in (0, 2, 4):
        s.walls_h[pr.slot_index(0, wc)] = 1
    # Vertical wall closing the right side of the pocket, leaving one escape.
    s.walls_v[pr.slot_index(0, 6)] = 1
    assert s.has_path(0), "escape route should still exist"
    legal = set(s.legal_wall_actions())
    # The wall that closes the last gap must be rejected.
    sealing = pr.wall_action(pr.HORIZONTAL, 0, 6)
    s2 = s.copy()
    s2._place_wall_unchecked(pr.HORIZONTAL, 0, 6)
    if not s2.has_path(0):
        assert sealing not in legal, "a fully-blocking wall must be illegal"
    assert np.array_equal(ref_mask(s), fast_mask(s))


def test_no_walls_left_means_no_wall_actions():
    s = pr.initial_state()
    s.walls_left[0] = 0
    assert s.legal_wall_actions() == []
    assert np.array_equal(ref_mask(s), fast_mask(s))
    assert fast_mask(s)[:128].sum() == 0


# --------------------------------------------------------- random playouts


@pytest.mark.parametrize("seed", range(12))
def test_random_playout_masks_agree(seed):
    """Play a random game, comparing the full legality mask at every ply."""
    rng = np.random.default_rng(seed)
    s = pr.initial_state()
    for ply in range(220):
        if s.is_terminal():
            break
        rm, fm = ref_mask(s), fast_mask(s)
        assert np.array_equal(rm, fm), (
            f"seed={seed} ply={ply} mismatch: {describe(rm, fm)}\n{s.render()}"
        )
        # Invariant: both players always retain a route to their goal.
        assert s.has_path(0) and s.has_path(1), f"seed={seed} ply={ply}: player sealed in"
        actions = np.nonzero(rm)[0]
        assert len(actions) > 0, "a player always has at least one legal move"
        s = s.apply(int(rng.choice(actions)))


@pytest.mark.parametrize("seed", range(6))
def test_apply_action_agrees(seed):
    """The fast core's apply_action must track the reference state exactly."""
    rng = np.random.default_rng(1000 + seed)
    s = pr.initial_state()
    st = fr.initial_state()
    scratch = fr.make_scratch()
    for _ in range(220):
        if s.is_terminal():
            break
        rm = ref_mask(s)
        actions = np.nonzero(rm)[0]
        a = int(rng.choice(actions))
        s = s.apply(a)
        fr.apply_action(st, a, scratch)
        assert np.array_equal(st, to_fast(s)), f"state diverged after action {a}"
    assert fr.winner(st) == (-1 if s.winner() is None else s.winner())


def test_random_playout_terminates_with_a_winner():
    """Random play should reliably produce a decisive game well inside the cap."""
    scratch = fr.make_scratch()
    wins = [0, 0]
    for seed in range(20):
        st = fr.initial_state()
        w = fr.random_playout(st, scratch, 1000, seed * 7919 + 13)
        assert w in (0, 1), f"random playout {seed} did not finish"
        wins[w] += 1
    assert sum(wins) == 20
