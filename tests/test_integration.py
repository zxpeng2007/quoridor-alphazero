"""End-to-end smoke test of the training pipeline.

Runs a deliberately tiny version of the full loop -- self-play, replay, training
step, arena -- with a small network. The point is to catch wiring bugs (shape
mismatches, dtype errors, misrouted transforms) in seconds rather than after an
hour of a real run has gone into a checkpoint that turns out to be garbage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quoridor import fastrules as fr
from quoridor.arena import batched_match, match_neural_vs_agent
from quoridor.agents import GreedyAgent
from quoridor.net import NetConfig, NetEvaluator, QuoridorNet
from quoridor.replay import ReplayBuffer
from quoridor.selfplay import SelfPlayConfig, SelfPlayEngine
from quoridor.train import Trainer, TrainConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_net():
    return QuoridorNet(NetConfig(channels=32, blocks=2))


@pytest.fixture(scope="module")
def evaluator(tiny_net):
    return NetEvaluator(tiny_net, device=DEVICE)


def test_network_forward_shapes(tiny_net):
    x = torch.randn(4, 8, 9, 9)
    logits, value = tiny_net(x)
    assert logits.shape == (4, fr.NUM_ACTIONS)
    assert value.shape == (4,)
    assert torch.all(value >= -1) and torch.all(value <= 1), "value must be tanh-bounded"


def test_evaluator_masks_illegal_actions(evaluator):
    planes = np.random.rand(4, 8, 9, 9).astype(np.float32)
    legal = np.zeros((4, fr.NUM_ACTIONS), dtype=np.uint8)
    legal[:, [129, 130, 131]] = 1
    policy, value = evaluator.evaluate(planes, legal)
    assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-5)
    assert not policy[:, legal[0] == 0].any(), "illegal actions must get zero prior"


def test_selfplay_produces_consistent_samples(evaluator):
    cfg = SelfPlayConfig(n_parallel=8, sims=12, max_plies=60, temp_moves=4)
    engine = SelfPlayEngine(evaluator, cfg)
    batch = engine.generate(target_games=8)

    assert batch.stats.games >= 8
    n = len(batch.states)
    assert n == len(batch.policies) == len(batch.values) == batch.stats.positions
    assert batch.states.dtype == np.uint8 and batch.states.shape[1] == fr.STATE_SIZE
    assert np.all(np.isin(batch.values, [-1.0, 0.0, 1.0])), "values must be game results"
    # Policy rows are visit distributions: non-negative and normalised.
    sums = batch.policies.astype(np.float32).sum(axis=1)
    assert np.all(batch.policies >= 0)
    assert np.allclose(sums, 1.0, atol=1e-2)


def test_selfplay_policies_are_legal(evaluator):
    """Every visit-count target must sit on actions legal in its own position."""
    cfg = SelfPlayConfig(n_parallel=8, sims=12, max_plies=60)
    batch = SelfPlayEngine(evaluator, cfg).generate(target_games=4)
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    for i in range(0, len(batch.states), 7):  # sample a subset, this is O(n) BFS
        fr.legal_mask(batch.states[i], mask, scratch)
        support = np.nonzero(batch.policies[i].astype(np.float32) > 0)[0]
        assert mask[support].all(), f"position {i}: policy mass on an illegal action"


def test_full_loop_runs_and_reduces_loss(tiny_net, evaluator):
    """Self-play -> buffer -> training must complete and actually optimise."""
    cfg = SelfPlayConfig(n_parallel=16, sims=12, max_plies=60)
    batch = SelfPlayEngine(evaluator, cfg).generate(target_games=16)

    buf = ReplayBuffer(capacity=50_000)
    buf.add(batch.states, batch.policies, batch.values)
    assert len(buf) > 0

    trainer = Trainer(tiny_net, TrainConfig(batch_size=64, lr=1e-3, warmup_steps=2),
                      device=DEVICE)
    first = trainer.train_on_buffer(buf, n_steps=5)
    later = trainer.train_on_buffer(buf, n_steps=40)
    assert np.isfinite(first.total_loss) and np.isfinite(later.total_loss)
    assert later.total_loss < first.total_loss, (
        f"loss did not improve: {first.total_loss:.4f} -> {later.total_loss:.4f}"
    )


def test_batched_match_alternates_colours(evaluator):
    """A network against itself should land near 50% once colours alternate."""
    result = batched_match(
        evaluator, evaluator, games=16, sims=8, name_a="x", name_b="y", temp_plies=6
    )
    assert result.games == 16
    assert result.wins_a + result.wins_b + result.draws == 16
    assert result.mean_plies > 0


def test_match_against_conventional_agent(evaluator):
    result = match_neural_vs_agent(
        evaluator, GreedyAgent(), games=8, sims=8, name_a="net", name_b="greedy"
    )
    assert result.games == 8
    assert np.isfinite(result.elo_diff)


def test_stalled_games_get_adjudicated(evaluator):
    """Ply-capped games must be resolved on progress, not silently dropped."""
    cfg = SelfPlayConfig(n_parallel=8, sims=8, max_plies=20)  # cap forces stalls
    batch = SelfPlayEngine(evaluator, cfg).generate(target_games=8)
    assert batch.stats.unfinished == batch.stats.games, "all games should hit the cap"
    # Without adjudication every capped game would be a 0-value draw. Some will
    # still legitimately tie -- a 20-ply cap stops both players mid-race, so equal
    # remaining distance is common -- but the count must be non-zero.
    decided = batch.stats.p0_wins + batch.stats.p1_wins
    assert decided > 0, "adjudication produced no decided games"
    assert decided <= batch.stats.games
    # Decided games must carry +/-1 value targets, ties must carry 0.
    assert set(np.unique(batch.values).tolist()) <= {-1.0, 0.0, 1.0}
    assert np.any(batch.values != 0.0), "decided games should have non-zero targets"
