"""Replay buffer and the batch encoder that feeds training.

Positions are stored as raw 133-byte states rather than encoded planes: the
encoding is 8x9x9 float32 = 2.6 KB, so keeping states instead cuts buffer memory
by ~20x and lets us hold millions of positions. Encoding happens per training
batch, parallelised across cores, which costs far less than the GPU step it feeds.

Two action-space transforms are applied per sample, in order:

1. **Mirror** (augmentation, random per sample) -- the only symmetry Quoridor
   admits, since the players' goal rows are fixed to the top and bottom.
2. **Canonicalisation** (mandatory) -- the network always sees the side-to-move
   view, so a player-1-to-move position is rotated 180 degrees.

Policy targets arrive in *real* action space and must follow both transforms,
otherwise the network is trained to play the right move on the wrong board.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from quoridor.encoding import (
    MIRROR_ACTION,
    NUM_PLANES,
    ROT180_ACTION,
    canonicalize,
    encode_canonical,
    mirror_state,
)
from quoridor.fastrules import NUM_ACTIONS, SCRATCH_SIZE, STATE_SIZE, legal_mask


@njit(cache=True, parallel=True)
def encode_training_batch(
    states, policies_in, policies_out, planes_out, legal_out, do_mirror,
    scratch_all, d0_all, d1_all, canon_all, tmp_state, work_policy,
    mirror_perm, rot,
):
    """Encode a sampled batch, applying mirror augmentation and canonicalisation.

    Also emits the legal-action mask for each canonical position. Training masks
    illegal actions out of the softmax exactly as inference does -- without it,
    late-game positions (where walls are spent and only a handful of the 140
    actions are legal) would spend most of their probability mass on moves that
    can never be played.
    """
    B = states.shape[0]
    for i in prange(B):
        # --- 1. optional mirror, on both the state and the policy target
        if do_mirror[i] != 0:
            mirror_state(states[i], tmp_state[i])
            for a in range(NUM_ACTIONS):
                work_policy[i, a] = policies_in[i, mirror_perm[a]]
        else:
            for k in range(STATE_SIZE):
                tmp_state[i, k] = states[i, k]
            for a in range(NUM_ACTIONS):
                work_policy[i, a] = policies_in[i, a]

        # --- 2. canonicalise into the side-to-move view
        flipped = canonicalize(tmp_state[i], canon_all[i])
        encode_canonical(
            canon_all[i], planes_out[i], scratch_all[i], d0_all[i], d1_all[i]
        )
        legal_mask(canon_all[i], legal_out[i], scratch_all[i])
        if flipped:
            for a in range(NUM_ACTIONS):
                policies_out[i, a] = work_policy[i, rot[a]]
        else:
            for a in range(NUM_ACTIONS):
                policies_out[i, a] = work_policy[i, a]


class ReplayBuffer:
    """Fixed-capacity ring buffer of self-play positions."""

    def __init__(self, capacity: int = 2_000_000, seed: int = 0):
        self.capacity = capacity
        self.states = np.zeros((capacity, STATE_SIZE), dtype=np.uint8)
        self.policies = np.zeros((capacity, NUM_ACTIONS), dtype=np.float16)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0
        self.total_added = 0
        self.rng = np.random.default_rng(seed)
        self._mirror_perm = MIRROR_ACTION.astype(np.int32)
        self._rot = ROT180_ACTION.astype(np.int32)
        self._bufs: dict[int, tuple] = {}

    def __len__(self) -> int:
        return self.size

    def _scratch_for(self, B: int) -> tuple:
        """Reuse per-batch-size scratch, so sampling does not allocate every call."""
        if B not in self._bufs:
            self._bufs[B] = (
                np.zeros((B, NUM_PLANES, 9, 9), dtype=np.float32),
                np.zeros((B, NUM_ACTIONS), dtype=np.float32),
                np.zeros((B, NUM_ACTIONS), dtype=np.uint8),
                np.zeros((B, SCRATCH_SIZE), dtype=np.int32),
                np.zeros((B, 81), dtype=np.int32),
                np.zeros((B, 81), dtype=np.int32),
                np.zeros((B, STATE_SIZE), dtype=np.uint8),
                np.zeros((B, STATE_SIZE), dtype=np.uint8),
                np.zeros((B, NUM_ACTIONS), dtype=np.float32),
            )
        return self._bufs[B]

    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        n = len(states)
        if n == 0:
            return
        if n >= self.capacity:  # keep only the newest `capacity` samples
            states = states[-self.capacity:]
            policies = policies[-self.capacity:]
            values = values[-self.capacity:]
            n = self.capacity
        end = self.pos + n
        if end <= self.capacity:
            sl = slice(self.pos, end)
            self.states[sl] = states
            self.policies[sl] = policies
            self.values[sl] = values
        else:
            first = self.capacity - self.pos
            self.states[self.pos:] = states[:first]
            self.policies[self.pos:] = policies[:first]
            self.values[self.pos:] = values[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.policies[:rest] = policies[first:]
            self.values[:rest] = values[first:]
        self.pos = end % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n

    def sample(self, batch_size: int, augment: bool = True):
        """Return ``(planes, policy, value, legal)`` for one training batch."""
        if self.size == 0:
            raise ValueError("replay buffer is empty")
        idx = self.rng.integers(0, self.size, size=batch_size)
        bufs = self._scratch_for(batch_size)
        planes, pol_out, legal_out, scratch, d0, d1, canon, tmp, work = bufs

        states = np.ascontiguousarray(self.states[idx])
        policies = self.policies[idx].astype(np.float32)
        values = self.values[idx].astype(np.float32)
        do_mirror = (
            self.rng.integers(0, 2, size=batch_size).astype(np.uint8)
            if augment
            else np.zeros(batch_size, dtype=np.uint8)
        )
        encode_training_batch(
            states, policies, pol_out, planes, legal_out, do_mirror,
            scratch, d0, d1, canon, tmp, work,
            self._mirror_perm, self._rot,
        )
        return planes, pol_out, values, legal_out

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            states=self.states[: self.size],
            policies=self.policies[: self.size],
            values=self.values[: self.size],
            total_added=self.total_added,
        )

    @classmethod
    def load(cls, path: str, capacity: int = 2_000_000) -> "ReplayBuffer":
        blob = np.load(path)
        buf = cls(capacity=capacity)
        buf.add(blob["states"], blob["policies"], blob["values"])
        buf.total_added = int(blob["total_added"]) if "total_added" in blob else buf.size
        return buf


class MixedBuffer:
    """Self-play positions plus an oversampled repair set, in fixed proportion.

    Mined positions number a few thousand against a two-million-position
    self-play buffer. Sampled uniformly they would reach the network only a
    handful of times before the ring evicts them -- nowhere near enough to move a
    value output that is wrong by 0.7. What decides whether a value error gets
    corrected is how often the network is shown the position, and that has to be
    set deliberately rather than left to the repair set's share of the buffer.

    Exposes the same ``sample`` contract as :class:`ReplayBuffer`, so the trainer
    needs no changes.
    """

    def __init__(self, main: ReplayBuffer, repair: ReplayBuffer,
                 frac: float = 0.25):
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"frac must be in [0, 1), got {frac}")
        self.main = main
        self.repair = repair
        self.frac = frac

    def __len__(self) -> int:
        return len(self.main)

    @property
    def total_added(self) -> int:
        return self.main.total_added

    def sample(self, batch_size: int, augment: bool = True):
        n_rep = int(round(batch_size * self.frac))
        if len(self.repair) == 0 or n_rep == 0:
            return self.main.sample(batch_size, augment)
        n_rep = min(n_rep, batch_size - 1)
        a = self.main.sample(batch_size - n_rep, augment)
        b = self.repair.sample(n_rep, augment)
        return tuple(np.concatenate([x, y], axis=0) for x, y in zip(a, b))
