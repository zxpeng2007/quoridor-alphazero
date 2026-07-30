"""Tests for the leaf-batched single-tree search.

Virtual-loss parallelism changes which leaves a batch explores, so exact
equality with the sequential search is not expected -- but the invariants that
make the search *sound* must hold: visit accounting, legality, tactical
correctness, and no leftover virtual-loss residue in the tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS
from quoridor.net import UniformEvaluator

WIN_S = fr.MOVE_BASE + 1


def position(p0: int, p1: int, turn: int = 0, walls: int = 10) -> np.ndarray:
    st = fr.initial_state()
    st[fr.IDX_P0] = p0
    st[fr.IDX_P1] = p1
    st[fr.IDX_TURN] = turn
    st[fr.IDX_WL0] = walls
    st[fr.IDX_WL1] = walls
    return st


@pytest.mark.parametrize("leaf_batch", [4, 32])
def test_visits_sum_to_sim_count(leaf_batch):
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=600,
                       leaf_batch=leaf_batch)
    visits = mcts.search(fr.initial_state().reshape(1, -1), 300)
    assert visits.shape == (1, fr.NUM_ACTIONS)
    assert visits.sum() == 300


@pytest.mark.parametrize("leaf_batch", [4, 32])
def test_no_virtual_loss_residue(leaf_batch):
    """After search, N must be non-negative integers: every provisional loss
    applied during selection must have been removed during backup."""
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=600,
                       leaf_batch=leaf_batch)
    mcts.search(fr.initial_state().reshape(1, -1), 256)
    used = mcts.node_count[0]
    n = mcts.node_N[0, :used]
    assert (n >= 0).all(), "negative visit count: virtual loss not fully removed"
    assert np.allclose(n, np.round(n)), "fractional visits: vloss residue"


def test_visits_only_on_legal_actions():
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=600, leaf_batch=16)
    st = fr.initial_state()
    visits = mcts.search(st.reshape(1, -1), 200)
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(st, mask, fr.make_scratch())
    assert not visits[0][mask == 0].any()


def test_finds_immediate_win_batched():
    st = position(fr.N * 7 + 4, fr.N * 1 + 0, walls=0)
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=600, leaf_batch=16)
    visits = mcts.search(st.reshape(1, -1).copy(), 200)
    assert int(visits[0].argmax()) == WIN_S
    assert mcts.root_value()[0] > 0.3


def test_recognises_lost_position_batched():
    st = position(fr.N * 1 + 4, fr.N * 1 + 0, walls=0)
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=800, leaf_batch=16)
    mcts.search(st.reshape(1, -1).copy(), 400)
    assert mcts.root_value()[0] < -0.3


def test_multi_game_instances_ignore_leaf_batch():
    """Self-play batches across games; leaf_batch must not engage there."""
    mcts = BatchedMCTS(UniformEvaluator(), n_games=4, max_nodes=300, leaf_batch=32)
    assert mcts.leaf_batch == 1
    states = np.stack([fr.initial_state() for _ in range(4)])
    visits = mcts.search(states, 60)
    assert np.allclose(visits.sum(axis=1), 60)


def test_batched_and_sequential_agree_on_easy_position():
    """On a position with one clearly best move the two searches must agree."""
    st = position(fr.N * 6 + 4, fr.N * 2 + 4, walls=0)  # pure race, P0 ahead
    seq = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=900)
    par = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=900, leaf_batch=16)
    v_seq = seq.search(st.reshape(1, -1).copy(), 400)
    v_par = par.search(st.reshape(1, -1).copy(), 400)
    assert int(v_seq[0].argmax()) == int(v_par[0].argmax())
