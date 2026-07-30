"""Self-play game generation.

Games are run in a fixed-size pool: whenever one finishes it is immediately
replaced by a fresh game in the same slot. Without that refill the batch drains
as games end at different lengths, and the last stragglers would run at a tiny
batch size with the GPU almost idle.

Training targets per position:

* **policy** -- the MCTS root visit distribution (the "improved" policy that
  search produces on top of the raw network prior)
* **value**  -- the eventual game result, from the perspective of the player to
  move in that position

Both are stored against the *raw* state; encoding into planes happens at training
time, since a 133-byte state is ~20x cheaper to keep in the replay buffer than
its 8x9x9 float32 encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quoridor import fastrules as fr
from quoridor.arena import decide_by_progress
from quoridor.mcts import BatchedMCTS


@dataclass
class SelfPlayConfig:
    n_parallel: int = 512
    sims: int = 200
    c_puct: float = 1.6
    temperature: float = 1.0
    temp_moves: int = 16  # plies of sampling before switching to greedy
    dirichlet_alpha: float = 0.15
    dirichlet_frac: float = 0.25
    max_plies: int = 250
    seed: int = 0


@dataclass
class SelfPlayStats:
    games: int = 0
    positions: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    unfinished: int = 0
    total_plies: int = 0

    @property
    def mean_plies(self) -> float:
        return self.total_plies / max(self.games, 1)

    @property
    def p0_win_rate(self) -> float:
        decided = self.p0_wins + self.p1_wins
        return self.p0_wins / max(decided, 1)

    def __str__(self) -> str:
        return (
            f"{self.games} games, {self.positions} positions, "
            f"{self.mean_plies:.1f} plies/game, "
            f"P0 win rate {100 * self.p0_win_rate:.1f}%"
            + (f", {self.unfinished} hit ply cap" if self.unfinished else "")
        )


@dataclass
class GameBatch:
    """Flat arrays of training samples produced by one generation run."""

    states: np.ndarray  # uint8[N, 133]
    policies: np.ndarray  # float16[N, 140]
    values: np.ndarray  # float32[N]
    stats: SelfPlayStats = field(default_factory=SelfPlayStats)


class SelfPlayEngine:
    """Generates self-play games with a fixed pool of concurrent boards."""

    def __init__(self, evaluator, cfg: SelfPlayConfig | None = None):
        self.cfg = cfg or SelfPlayConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.mcts = BatchedMCTS(
            evaluator,
            n_games=self.cfg.n_parallel,
            max_nodes=self.cfg.sims + 64,
            c_puct=self.cfg.c_puct,
            dirichlet_alpha=self.cfg.dirichlet_alpha,
            dirichlet_frac=self.cfg.dirichlet_frac,
            seed=self.cfg.seed,
        )
        self.scratch = fr.make_scratch()

    def generate(self, target_games: int, progress: bool = False) -> GameBatch:
        """Play until ``target_games`` games have finished; return their samples."""
        cfg = self.cfg
        G = cfg.n_parallel

        states = np.stack([fr.initial_state() for _ in range(G)])
        # Per-slot in-progress history: (state, policy, player_to_move)
        hist_states: list[list[np.ndarray]] = [[] for _ in range(G)]
        hist_policy: list[list[np.ndarray]] = [[] for _ in range(G)]
        hist_player: list[list[int]] = [[] for _ in range(G)]
        ply = np.zeros(G, dtype=np.int32)

        out_states: list[np.ndarray] = []
        out_policy: list[np.ndarray] = []
        out_value: list[np.ndarray] = []
        stats = SelfPlayStats()

        bar = None
        if progress:
            from tqdm import tqdm

            bar = tqdm(total=target_games, desc="self-play", unit="game")

        while stats.games < target_games:
            visits = self.mcts.search(states, cfg.sims, add_noise=True)

            for g in range(G):
                total = visits[g].sum()
                if total <= 0:  # terminal root; should have been reset already
                    continue
                policy = visits[g] / total
                player = int(states[g][fr.IDX_TURN])

                hist_states[g].append(states[g].copy())
                hist_policy[g].append(policy.astype(np.float16))
                hist_player[g].append(player)

                if ply[g] < cfg.temp_moves and cfg.temperature > 1e-3:
                    p = np.power(policy, 1.0 / cfg.temperature)
                    p = p / p.sum()
                    action = int(self.rng.choice(len(p), p=p))
                else:
                    action = int(policy.argmax())

                fr.apply_action(states[g], action, self.scratch)
                ply[g] += 1

                w = fr.winner(states[g])
                finished = w >= 0
                capped = ply[g] >= cfg.max_plies
                if not (finished or capped):
                    continue

                # Score the finished game and flush its positions. A game that
                # ran out of plies is adjudicated on remaining distance rather
                # than discarded -- see arena.decide_by_progress for why.
                result = w
                if not finished:
                    stats.unfinished += 1
                    result = decide_by_progress(states[g], self.scratch)
                stats.p0_wins += result == 0
                stats.p1_wins += result == 1

                n = len(hist_states[g])
                vals = np.zeros(n, dtype=np.float32)
                if result >= 0:
                    for i, mover in enumerate(hist_player[g]):
                        vals[i] = 1.0 if mover == result else -1.0
                out_states.extend(hist_states[g])
                out_policy.extend(hist_policy[g])
                out_value.append(vals)

                stats.games += 1
                stats.positions += n
                stats.total_plies += int(ply[g])
                if bar is not None:
                    bar.update(1)

                hist_states[g].clear()
                hist_policy[g].clear()
                hist_player[g].clear()
                ply[g] = 0
                states[g] = fr.initial_state()

        if bar is not None:
            bar.close()

        return GameBatch(
            states=np.stack(out_states) if out_states else np.zeros((0, fr.STATE_SIZE), np.uint8),
            policies=np.stack(out_policy) if out_policy else np.zeros((0, fr.NUM_ACTIONS), np.float16),
            values=np.concatenate(out_value) if out_value else np.zeros(0, np.float32),
            stats=stats,
        )
