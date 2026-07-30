"""Tests for batched PUCT MCTS.

The sign convention on backed-up values is the easiest thing in an MCTS to get
backwards, and a sign error does not crash -- it just trains a network to prefer
losing. So the tests here pin down that search *prefers winning moves and avoids
losing ones*, rather than only checking shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS, MCTSAgent, visits_to_policy
from quoridor.net import UniformEvaluator

WIN_S = fr.MOVE_BASE + 1  # action 129: step south


def position(p0: int, p1: int, turn: int = 0, walls: int = 10) -> np.ndarray:
    st = fr.initial_state()
    st[fr.IDX_P0] = p0
    st[fr.IDX_P1] = p1
    st[fr.IDX_TURN] = turn
    st[fr.IDX_WL0] = walls
    st[fr.IDX_WL1] = walls
    return st


def cell(r: int, c: int) -> int:
    return r * 9 + c


# ------------------------------------------------------------------- basics


def test_root_visits_sum_to_simulation_count():
    mcts = BatchedMCTS(UniformEvaluator(), n_games=4, max_nodes=400)
    states = np.stack([fr.initial_state() for _ in range(4)])
    visits = mcts.search(states, n_sims=100)
    assert visits.shape == (4, fr.NUM_ACTIONS)
    assert np.allclose(visits.sum(axis=1), 100)


def test_visits_only_on_legal_actions():
    mcts = BatchedMCTS(UniformEvaluator(), n_games=2, max_nodes=400)
    states = np.stack([fr.initial_state() for _ in range(2)])
    visits = mcts.search(states, n_sims=120)
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(states[0], mask, scratch)
    assert not visits[0][mask == 0].any(), "search visited an illegal action"


# ---------------------------------------------------------- tactical checks


# The tactical tests below set walls to zero deliberately. With walls in hand a
# position has ~130 legal actions, and a uniform prior forces PUCT to try every
# one before revisiting anything -- so a knowledge-free search cannot resolve
# even a 2-ply tactic within a sane simulation budget. Removing walls cuts the
# branching factor to ~4 and isolates the value logic, which is what these tests
# are actually about. (Search with a *trained* prior does not have this problem;
# that is the whole point of the policy head.)


def test_finds_immediate_win():
    """One step from the goal, search must overwhelmingly pick the winning move."""
    st = position(cell(7, 4), cell(1, 0), turn=0, walls=0)
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=400)
    visits = mcts.search(st.reshape(1, -1).copy(), n_sims=200)
    assert int(visits[0].argmax()) == WIN_S
    assert visits[0][WIN_S] > 0.5 * visits[0].sum(), "winning move should dominate"


def test_root_value_is_positive_when_winning():
    """A won position must evaluate positively for the side to move."""
    st = position(cell(7, 4), cell(1, 0), turn=0, walls=0)
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=400)
    mcts.search(st.reshape(1, -1).copy(), n_sims=200)
    assert mcts.root_value()[0] > 0.3, "value backup sign looks inverted"


def test_avoids_handing_opponent_the_win():
    """With the opponent one step from home, search should evaluate this as lost."""
    st = position(cell(1, 4), cell(1, 0), turn=0, walls=0)
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=600)
    mcts.search(st.reshape(1, -1).copy(), n_sims=400)
    # Opponent steps north to row 0 and wins whatever we do.
    assert mcts.root_value()[0] < -0.3, "should recognise a lost position"


def test_uniform_prior_must_sweep_wide_action_space():
    """Documents the branching problem: with walls, the win is found late.

    This is not a defect in the search -- it is why the policy prior matters so
    much in Quoridor, and it is worth having pinned down so a future change to
    the action ordering does not quietly make it worse.
    """
    st = position(cell(7, 4), cell(1, 0), turn=0, walls=10)
    mcts = BatchedMCTS(
        UniformEvaluator(), n_games=1, max_nodes=400, fpu_reduction=0.0
    )
    visits = mcts.search(st.reshape(1, -1).copy(), n_sims=200)
    assert int(visits[0].argmax()) == WIN_S, "the win is still found eventually"
    others = np.delete(visits[0], WIN_S)
    assert others.max() <= 1.0, "every other action explored exactly once"


def test_fpu_reduction_concentrates_search():
    """FPU reduction must trade breadth for depth.

    Under a uniform prior this is actively harmful -- it collapses search onto
    the first action tried -- which is exactly why the default is tuned for a
    trained prior and the bootstrap stage exists. Pinned down here so the
    trade-off stays visible.
    """
    st = position(cell(7, 4), cell(1, 0), turn=0, walls=10)
    wide = BatchedMCTS(
        UniformEvaluator(), n_games=1, max_nodes=400, fpu_reduction=0.0
    ).search(st.reshape(1, -1).copy(), n_sims=200)
    narrow = BatchedMCTS(
        UniformEvaluator(), n_games=1, max_nodes=400, fpu_reduction=0.4
    ).search(st.reshape(1, -1).copy(), n_sims=200)

    n_narrow, n_wide = (narrow > 0).sum(), (wide > 0).sum()
    assert n_narrow < n_wide, "FPU reduction should visit fewer distinct moves"
    # Depth is measured as visits per explored move, not max visits: the wide
    # search stumbles onto the terminal win and pours everything into it, which
    # would make a max-visit comparison misleading.
    assert narrow.sum() / n_narrow > wide.sum() / n_wide, (
        "FPU reduction should search each retained move more deeply"
    )


def test_search_handles_terminal_root():
    """A root that is already decided must not crash or produce visits."""
    st = position(cell(8, 4), cell(1, 0), turn=1)  # player 0 already home
    assert fr.winner(st) == 0
    mcts = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=200)
    visits = mcts.search(st.reshape(1, -1).copy(), n_sims=50)
    assert visits.sum() == 0


# ------------------------------------------------------------ batch equality


def test_batched_search_matches_single_game():
    """Games in a batch must not influence each other."""
    st = fr.initial_state()
    single = BatchedMCTS(UniformEvaluator(), n_games=1, max_nodes=400)
    v_single = single.search(st.reshape(1, -1).copy(), n_sims=150)

    batched = BatchedMCTS(UniformEvaluator(), n_games=8, max_nodes=400)
    states = np.stack([st.copy() for _ in range(8)])
    v_batch = batched.search(states, n_sims=150)

    for g in range(8):
        assert np.array_equal(v_batch[g], v_single[0]), f"game {g} diverged"


def test_independent_games_stay_independent():
    """Different root positions in one batch must produce different searches."""
    a = fr.initial_state()
    b = position(cell(7, 4), cell(1, 0), turn=0)
    mcts = BatchedMCTS(
        UniformEvaluator(), n_games=2, max_nodes=400, fpu_reduction=0.0
    )
    visits = mcts.search(np.stack([a, b]), n_sims=200)
    assert int(visits[1].argmax()) == WIN_S
    assert int(visits[0].argmax()) != WIN_S or visits[0][WIN_S] < visits[1][WIN_S]


# ----------------------------------------------------------------- policies


def test_visits_to_policy_temperatures():
    visits = np.array([[10.0, 30.0, 0.0, 60.0]])
    greedy = visits_to_policy(visits, 0.0)
    assert greedy.tolist() == [[0.0, 0.0, 0.0, 1.0]]

    soft = visits_to_policy(visits, 1.0)
    assert soft.sum() == pytest.approx(1.0)
    assert soft[0].tolist() == pytest.approx([0.1, 0.3, 0.0, 0.6])

    flat = visits_to_policy(visits, 10.0)
    assert flat[0][3] < soft[0][3], "high temperature should flatten the policy"


def test_mcts_agent_returns_legal_actions():
    agent = MCTSAgent(UniformEvaluator(), n_sims=60, max_nodes=300)
    st = fr.initial_state()
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    for _ in range(12):
        if fr.winner(st) >= 0:
            break
        a = agent.select_action(st)
        fr.legal_mask(st, mask, scratch)
        assert mask[a] == 1, f"agent chose illegal action {a}"
        fr.apply_action(st, a, scratch)
