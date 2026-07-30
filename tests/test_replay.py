"""Tests for the replay buffer and its augmentation/canonicalisation pipeline.

A transform bug here is invisible at runtime -- training simply converges to
playing good moves on a mirrored or rotated board. So the encoder is checked
against an independent Python reference, and the policy target's support is
checked to land on actually-legal actions.
"""

from __future__ import annotations

import numpy as np
import pytest

from quoridor import encoding as enc
from quoridor import fastrules as fr
from quoridor.replay import ReplayBuffer, encode_training_batch


def random_positions(n: int = 40) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    out: list[np.ndarray] = []
    while len(out) < n:
        st = fr.initial_state()
        for _ in range(rng.integers(1, 60)):
            if fr.winner(st) >= 0:
                break
            fr.legal_mask(st, mask, scratch)
            acts = np.nonzero(mask)[0]
            if not len(acts):
                break
            fr.apply_action(st, int(rng.choice(acts)), scratch)
        if fr.winner(st) < 0:
            out.append(st.copy())
    return out


def legal_of(st: np.ndarray) -> np.ndarray:
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(st, mask, fr.make_scratch())
    return mask


def reference_transform(state, policy, mirror):
    """Independent reference for the encoder, built from the encoding module."""
    s = state
    p = policy.copy()
    if mirror:
        s2 = np.empty(fr.STATE_SIZE, dtype=np.uint8)
        enc.mirror_state(s, s2)
        s = s2
        p = p[enc.MIRROR_ACTION]
    _, flipped = enc.to_canonical(s)
    planes = enc.encode(s)
    if flipped:
        p = p[enc.ROT180_ACTION]
    return planes, p


def run_encoder(states, policies, do_mirror):
    B = len(states)
    planes = np.zeros((B, enc.NUM_PLANES, 9, 9), dtype=np.float32)
    pol_out = np.zeros((B, fr.NUM_ACTIONS), dtype=np.float32)
    legal_out = np.zeros((B, fr.NUM_ACTIONS), dtype=np.uint8)
    encode_training_batch(
        np.ascontiguousarray(states), policies.astype(np.float32), pol_out, planes,
        legal_out, do_mirror.astype(np.uint8),
        np.zeros((B, fr.SCRATCH_SIZE), dtype=np.int32),
        np.zeros((B, 81), dtype=np.int32),
        np.zeros((B, 81), dtype=np.int32),
        np.zeros((B, fr.STATE_SIZE), dtype=np.uint8),
        np.zeros((B, fr.STATE_SIZE), dtype=np.uint8),
        np.zeros((B, fr.NUM_ACTIONS), dtype=np.float32),
        enc.MIRROR_ACTION.astype(np.int32),
        enc.ROT180_ACTION.astype(np.int32),
    )
    return planes, pol_out, legal_out


# ---------------------------------------------------------------- encoder


@pytest.mark.parametrize("mirror", [False, True])
def test_encoder_matches_reference(mirror):
    positions = random_positions()
    rng = np.random.default_rng(5)
    policies = rng.random((len(positions), fr.NUM_ACTIONS)).astype(np.float32)
    policies /= policies.sum(axis=1, keepdims=True)
    states = np.stack(positions)
    do_mirror = np.full(len(positions), int(mirror), dtype=np.uint8)

    planes, pol, legal = run_encoder(states, policies, do_mirror)
    for i, st in enumerate(positions):
        ref_planes, ref_pol = reference_transform(st, policies[i], mirror)
        assert np.allclose(planes[i], ref_planes), f"planes differ at {i}"
        assert np.allclose(pol[i], ref_pol), f"policy differs at {i}"


@pytest.mark.parametrize("mirror", [False, True])
def test_emitted_legal_mask_matches_canonical_position(mirror):
    """The mask handed to the loss must match the board the planes describe."""
    positions = random_positions()
    states = np.stack(positions)
    policies = np.zeros((len(positions), fr.NUM_ACTIONS), dtype=np.float32)
    policies[:, 0] = 1.0
    do_mirror = np.full(len(positions), int(mirror), dtype=np.uint8)

    _, _, legal = run_encoder(states, policies, do_mirror)
    for i, st in enumerate(positions):
        s = st
        if mirror:
            s2 = np.empty(fr.STATE_SIZE, dtype=np.uint8)
            enc.mirror_state(s, s2)
            s = s2
        canon, _ = enc.to_canonical(s)
        assert np.array_equal(legal[i], legal_of(canon)), f"legal mask differs at {i}"


def test_policy_support_stays_legal():
    """After both transforms, probability mass must sit only on legal actions."""
    positions = random_positions()
    states = np.stack(positions)
    # One-hot policy on a genuinely legal action for each position.
    policies = np.zeros((len(positions), fr.NUM_ACTIONS), dtype=np.float32)
    chosen = []
    rng = np.random.default_rng(9)
    for i, st in enumerate(positions):
        legal = np.nonzero(legal_of(st))[0]
        a = int(rng.choice(legal))
        policies[i, a] = 1.0
        chosen.append(a)

    for mirror in (False, True):
        do_mirror = np.full(len(positions), int(mirror), dtype=np.uint8)
        _, pol, _ = run_encoder(states, policies, do_mirror)
        for i, st in enumerate(positions):
            # Rebuild the transformed board and check the chosen action is legal there.
            s = st
            if mirror:
                s2 = np.empty(fr.STATE_SIZE, dtype=np.uint8)
                enc.mirror_state(s, s2)
                s = s2
            canon, _ = enc.to_canonical(s)
            a_out = int(pol[i].argmax())
            assert pol[i].sum() == pytest.approx(1.0)
            assert legal_of(canon)[a_out] == 1, (
                f"position {i}: transformed policy points at illegal action {a_out}"
            )


def test_mirror_preserves_policy_mass():
    positions = random_positions()
    states = np.stack(positions)
    rng = np.random.default_rng(2)
    policies = rng.random((len(positions), fr.NUM_ACTIONS)).astype(np.float32)
    policies /= policies.sum(axis=1, keepdims=True)
    _, pol, _ = run_encoder(states, policies, np.ones(len(positions), dtype=np.uint8))
    assert np.allclose(pol.sum(axis=1), 1.0)


# ----------------------------------------------------------------- buffer


def test_buffer_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=1000)
    positions = random_positions(30)
    states = np.stack(positions)
    policies = np.full((30, fr.NUM_ACTIONS), 1.0 / fr.NUM_ACTIONS, dtype=np.float16)
    values = np.ones(30, dtype=np.float32)
    buf.add(states, policies, values)
    assert len(buf) == 30

    planes, pol, val, legal = buf.sample(16)
    assert planes.shape == (16, enc.NUM_PLANES, 9, 9)
    assert pol.shape == (16, fr.NUM_ACTIONS)
    assert val.shape == (16,)
    assert legal.shape == (16, fr.NUM_ACTIONS)
    assert legal.sum() > 0, "sampled positions should have legal moves"
    # Policies are stored as float16 to keep the buffer small, so a 140-way
    # uniform distribution round-trips to ~0.99976 rather than exactly 1.
    assert np.allclose(pol.sum(axis=1), 1.0, atol=1e-3)


def test_buffer_ring_wraps_and_keeps_newest():
    buf = ReplayBuffer(capacity=10)
    for tag in range(3):
        states = np.zeros((6, fr.STATE_SIZE), dtype=np.uint8)
        states[:, 0] = tag  # marker we can look for later
        buf.add(
            states,
            np.zeros((6, fr.NUM_ACTIONS), dtype=np.float16),
            np.full(6, float(tag), dtype=np.float32),
        )
    assert len(buf) == 10
    # 18 added into capacity 10: only the newest 10 survive (tags 1 and 2).
    assert set(np.unique(buf.values).tolist()) <= {1.0, 2.0}


def test_buffer_rejects_empty_sample():
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buf.sample(4)
