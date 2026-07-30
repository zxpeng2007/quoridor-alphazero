"""Tests for the GUI game session.

Uses the prior-free UniformEvaluator so no checkpoint or GPU is needed; the
session logic under test (turn order, legality, undo, coaching, serialization)
is identical either way.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quoridor import fastrules as fr
from quoridor.net import UniformEvaluator
from quoridor.webgui import LEVELS, GameSession, action_text, cell_name


@pytest.fixture
def session():
    s = GameSession(UniformEvaluator(), seed=1)
    # Keep the coach searches cheap for tests.
    s.eval_sims = 16
    s.hint_sims = 16
    s.new_game(human_player=0, level="beginner")
    return s


def play(session, action):
    """One full exchange: the human's move plus the engine's (separate) reply."""
    session.human_move(action)
    if session.result is None:
        session.engine_reply()


# ----------------------------------------------------------------- basics


def test_new_game_human_first(session):
    snap = session.snapshot()
    assert snap["ply"] == 0
    assert snap["turn"] == 0 == snap["human_player"]
    assert snap["result"] is None
    assert len(snap["legal_moves"]) == 3  # S, E, W from the start
    assert len(snap["legal_walls"]) == 128


def test_new_game_engine_first_defers_reply(session):
    """The engine's opening move is NOT played inside new_game: the client asks
    for it separately so the fresh board renders before the engine thinks."""
    session.new_game(human_player=1, level="beginner")
    snap = session.snapshot()
    assert snap["ply"] == 0
    assert snap["needs_reply"] is True
    session.engine_reply()
    snap = session.snapshot()
    assert snap["ply"] == 1
    assert snap["turn"] == 1 == snap["human_player"]
    assert snap["needs_reply"] is False
    assert snap["log"][0]["actor"] == "engine"


def test_human_move_returns_before_engine_reply(session):
    session.human_move(129)  # S -- must return with only the human move applied
    snap = session.snapshot()
    assert snap["ply"] == 1
    assert snap["turn"] == 1, "engine to move"
    assert snap["needs_reply"] is True
    assert snap["log"][0]["eval"] is None, "eval settles when the reply lands"
    assert snap["legal_moves"] == [], "no input while the engine owes a move"

    session.engine_reply()
    snap = session.snapshot()
    assert snap["ply"] == 2
    assert snap["turn"] == 0, "after the engine reply it is the human's turn again"
    assert [e["actor"] for e in snap["log"]] == ["you", "engine"]
    assert snap["log"][0]["eval"] is not None, "coach mode should record an eval"


def test_engine_reply_is_idempotent(session):
    """Duplicate /api/reply (double request, refresh mid-reply) must be a no-op."""
    session.engine_reply()  # human to move: nothing should happen
    assert session.snapshot()["ply"] == 0
    play(session, 129)
    before = session.snapshot()
    session.engine_reply()  # human to move again: still a no-op
    assert session.snapshot()["ply"] == before["ply"]


def test_illegal_moves_rejected(session):
    with pytest.raises(ValueError):
        session.human_move(128)  # N from (0,4) is off the board
    play(session, 129)
    with pytest.raises(ValueError):
        session.human_move(9999)


def test_wall_move_spends_a_wall(session):
    session.human_move(0)  # wall H(0,0) -- legal in the opening
    snap = session.snapshot()
    assert snap["walls_left"][0] == 9
    assert snap["walls_h"][0] == 1


# ------------------------------------------------------------------- undo


def test_takeback_returns_to_human_turn(session):
    play(session, 129)
    play(session, 129)
    assert session.snapshot()["ply"] == 4
    session.takeback()
    snap = session.snapshot()
    assert snap["ply"] == 2
    assert snap["turn"] == 0
    session.takeback()
    assert session.snapshot()["ply"] == 0
    with pytest.raises(ValueError):
        session.takeback()


def test_takeback_after_game_over(session):
    session.state[fr.IDX_P0] = fr.N * 7 + 4  # one step from the goal row
    session.state[fr.IDX_P1] = fr.N * 8 + 0  # out of the way (S would otherwise be a jump)
    session.history[-1] = session.state.copy()
    session.human_move(129)  # S -> reaches row 8, human wins
    assert session.snapshot()["result"] == 0
    session.takeback()
    snap = session.snapshot()
    assert snap["result"] is None
    assert snap["turn"] == 0


# ----------------------------------------------------------------- results


def test_human_win_detected(session):
    session.state[fr.IDX_P0] = fr.N * 7 + 4
    session.state[fr.IDX_P1] = fr.N * 8 + 0  # out of the way (S would otherwise be a jump)
    session.history[-1] = session.state.copy()
    session.human_move(129)
    snap = session.snapshot()
    assert snap["result"] == 0
    assert snap["log"][-1]["eval"] == 1.0
    assert snap["legal_moves"] == [] and snap["legal_walls"] == []
    with pytest.raises(ValueError):
        session.human_move(129)


def test_resign(session):
    session.resign()
    snap = session.snapshot()
    assert snap["result"] == 1 and snap["resigned"] is True
    with pytest.raises(ValueError):
        session.resign()


# ------------------------------------------------------------------- coach


def test_hint_returns_legal_top_moves(session):
    hint = session.hint()
    assert -1.0 <= hint["eval"] <= 1.0
    assert 1 <= len(hint["top"]) <= 5
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(session.state, mask, session.scratch)
    for entry in hint["top"]:
        assert mask[entry["action"]] == 1, "hint suggested an illegal action"
        if entry["action"] >= fr.MOVE_BASE:
            assert entry["dest"] is not None


def test_coach_off_skips_pre_eval(session):
    session.set_config(coach=False)
    play(session, 129)
    entry = session.snapshot()["log"][0]
    assert entry["tag"] is None and entry["best"] is None
    assert entry["eval"] is not None, "the reply search still provides an eval"


def test_level_change_mid_game(session):
    session.set_config(level="easy")
    assert session.snapshot()["level"] == "easy"
    with pytest.raises(ValueError):
        session.set_config(level="nope")


# ------------------------------------------------------------ serialization


def test_snapshot_is_json_serializable(session):
    play(session, 129)
    text = json.dumps(session.snapshot())
    assert "legal_moves" in text


def test_engine_moves_are_legal_over_a_full_game(session):
    """Play the engine against itself via the session API; every move must apply."""
    session.new_game(human_player=0, level="beginner")
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    for _ in range(120):
        snap = session.snapshot()
        if snap["result"] is not None:
            break
        moves = snap["legal_moves"]
        assert moves, "human should always have a pawn move"
        # Prefer advancing so games finish inside the loop budget.
        best = min(moves, key=lambda m: abs(m["dest"] // 9 - 8))
        play(session, best["action"])
    assert session.snapshot()["ply"] > 4


# ------------------------------------------------------------------ naming


def test_names():
    assert cell_name(0) == "a1"
    assert cell_name(80) == "i9"
    assert cell_name(4) == "e1"
    assert action_text(0) == "═ a1"
    assert action_text(64 + 7 * 8 + 7) == "║ h8"
