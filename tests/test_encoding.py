"""Tests for canonicalisation, symmetries, and network input encoding.

The equivariance tests are the important ones. A wrong entry in an action
permutation would not crash anything -- it would just quietly teach the network
the wrong move for mirrored positions -- so we check that the symmetries commute
with both *legality* and *dynamics* over real positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from quoridor import encoding as enc
from quoridor import fastrules as fr


def random_positions(n_games: int = 8, sample_rate: float = 0.35) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    out: list[np.ndarray] = []
    for _ in range(n_games):
        st = fr.initial_state()
        for _ply in range(200):
            if fr.winner(st) >= 0:
                break
            fr.legal_mask(st, mask, scratch)
            acts = np.nonzero(mask)[0]
            if len(acts) == 0:
                break
            if rng.random() < sample_rate:
                out.append(st.copy())
            fr.apply_action(st, int(rng.choice(acts)), scratch)
    return out


POSITIONS = random_positions()


def legal_of(st: np.ndarray) -> np.ndarray:
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(st, mask, fr.make_scratch())
    return mask


# ------------------------------------------------------------ permutations


def test_action_permutations_are_involutions():
    rot, mir = enc.ROT180_ACTION, enc.MIRROR_ACTION
    idx = np.arange(fr.NUM_ACTIONS)
    assert np.array_equal(rot[rot], idx), "180-degree rotation must be self-inverse"
    assert np.array_equal(mir[mir], idx), "mirror must be self-inverse"
    assert sorted(rot.tolist()) == idx.tolist(), "rotation must be a permutation"
    assert sorted(mir.tolist()) == idx.tolist(), "mirror must be a permutation"


def test_permutations_preserve_action_kinds():
    """Walls must map to walls of the same orientation; moves to moves."""
    for perm in (enc.ROT180_ACTION, enc.MIRROR_ACTION):
        for a in range(fr.NUM_ACTIONS):
            b = int(perm[a])
            assert (a >= fr.MOVE_BASE) == (b >= fr.MOVE_BASE)
            if a < fr.MOVE_BASE:
                assert (a // fr.NUM_WALL_SLOTS) == (b // fr.NUM_WALL_SLOTS)


# ------------------------------------------------------------ state symmetry


def test_rot180_is_involution():
    tmp = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    back = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    for st in POSITIONS:
        enc.rot180_state(st, tmp)
        enc.rot180_state(tmp, back)
        assert np.array_equal(st, back)


def test_mirror_is_involution():
    tmp = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    back = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    for st in POSITIONS:
        enc.mirror_state(st, tmp)
        enc.mirror_state(tmp, back)
        assert np.array_equal(st, back)


# ------------------------------------------------------------- equivariance


def test_rotation_commutes_with_legality():
    tmp = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    for st in POSITIONS:
        enc.rot180_state(st, tmp)
        base, rot = legal_of(st), legal_of(tmp)
        assert np.array_equal(rot[enc.ROT180_ACTION], base), "rotated legality mismatch"


def test_mirror_commutes_with_legality():
    tmp = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    for st in POSITIONS:
        enc.mirror_state(st, tmp)
        base, mir = legal_of(st), legal_of(tmp)
        assert np.array_equal(mir[enc.MIRROR_ACTION], base), "mirrored legality mismatch"


@pytest.mark.parametrize("which", ["rot", "mirror"])
def test_symmetry_commutes_with_dynamics(which):
    """Transform-then-move must equal move-then-transform."""
    scratch = fr.make_scratch()
    transform = enc.rot180_state if which == "rot" else enc.mirror_state
    perm = enc.ROT180_ACTION if which == "rot" else enc.MIRROR_ACTION
    rng = np.random.default_rng(3)
    tmp = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    tmp2 = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    for st in POSITIONS:
        legal = np.nonzero(legal_of(st))[0]
        a = int(rng.choice(legal))

        moved = st.copy()
        fr.apply_action(moved, a, scratch)
        transform(moved, tmp)  # move, then transform

        transform(st, tmp2)
        fr.apply_action(tmp2, int(perm[a]), scratch)  # transform, then move

        assert np.array_equal(tmp, tmp2), f"{which} does not commute with action {a}"


# ------------------------------------------------------------ canonical form


def test_canonicalize_always_yields_side_to_move_zero():
    out = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    saw_flip = False
    for st in POSITIONS:
        flipped = enc.canonicalize(st, out)
        assert out[fr.IDX_TURN] == 0
        assert flipped == (st[fr.IDX_TURN] == 1)
        saw_flip |= bool(flipped)
    assert saw_flip, "sample should include player-1-to-move positions"


def test_canonical_legality_matches_via_permutation():
    """Legal actions in canonical space map back to the real board correctly."""
    for st in POSITIONS:
        canon, flipped = enc.to_canonical(st)
        base, canon_mask = legal_of(st), legal_of(canon)
        if flipped:
            canon_mask = canon_mask[enc.ROT180_ACTION]
        assert np.array_equal(canon_mask, base)


def test_decanonicalize_action_roundtrip():
    for st in POSITIONS:
        canon, flipped = enc.to_canonical(st)
        for a in np.nonzero(legal_of(canon))[0]:
            real = enc.decanonicalize_action(int(a), flipped)
            assert legal_of(st)[real] == 1, "decanonicalised action must be legal"


# ----------------------------------------------------------------- encoding


def test_encode_shape_and_range():
    for st in POSITIONS[:20]:
        planes = enc.encode(st)
        assert planes.shape == (enc.NUM_PLANES, 9, 9)
        assert planes.dtype == np.float32
        assert planes.min() >= 0.0 and planes.max() <= 1.0
        # Exactly one pawn per pawn-plane.
        assert planes[enc.PLANE_SELF_PAWN].sum() == pytest.approx(1.0)
        assert planes[enc.PLANE_OPP_PAWN].sum() == pytest.approx(1.0)


def test_encode_wall_planes_match_state():
    for st in POSITIONS[:20]:
        canon, _ = enc.to_canonical(st)
        planes = enc.encode(st)
        wh = np.asarray(canon[fr.WH_OFF:fr.WH_OFF + 64], dtype=np.float32).reshape(8, 8)
        wv = np.asarray(canon[fr.WV_OFF:fr.WV_OFF + 64], dtype=np.float32).reshape(8, 8)
        assert np.array_equal(planes[enc.PLANE_WALLS_H][:8, :8], wh)
        assert np.array_equal(planes[enc.PLANE_WALLS_V][:8, :8], wv)
        assert planes[enc.PLANE_WALLS_H][8, :].sum() == 0  # padding row stays empty
        assert planes[enc.PLANE_WALLS_H][:, 8].sum() == 0


def test_encoded_distance_plane_matches_shortest_path():
    scratch = fr.make_scratch()
    for st in POSITIONS[:20]:
        canon, _ = enc.to_canonical(st)
        planes = enc.encode(st)
        p0 = canon[fr.IDX_P0]
        d = fr.shortest_path_len(canon, 0, scratch)
        expected = 1.0 if d >= fr.INF else min(d, 40) / enc.MAX_DIST
        assert planes[enc.PLANE_SELF_DIST][p0 // 9, p0 % 9] == pytest.approx(expected)


def test_mirror_planes_matches_mirrored_state():
    """The plane-level mirror must agree with mirroring the state then encoding."""
    tmp = np.empty(fr.STATE_SIZE, dtype=np.uint8)
    for st in POSITIONS[:20]:
        enc.mirror_state(st, tmp)
        assert np.allclose(enc.encode(tmp), enc.mirror_planes(enc.encode(st))), (
            "mirror_planes disagrees with mirror_state + encode"
        )
