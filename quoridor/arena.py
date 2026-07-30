"""Match play and Elo estimation.

Everything the project claims about strength routes through here: promotion
gating during training, baseline comparisons, and the final strength report.
Colours are always alternated, because Quoridor has a real first-player
advantage and a same-colour match would badly misreport a difference.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from quoridor import fastrules as fr

AgentFactory = Callable[[], "object"]

DRAW = -1
MAX_PLIES = 300

# Virtual draws added to every pairing before computing Elo. Without this a
# shutout is a complete-separation case: the maximum-likelihood Elo difference
# is infinite, which is how a 40-0 result turned into "+4022 Elo +/- 1702434".
PRIOR_GAMES = 2.0


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    wins_a: int
    wins_b: int
    draws: int
    mean_plies: float

    @property
    def games(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    @property
    def score_a(self) -> float:
        """Points per game for A, counting a draw as half."""
        return (self.wins_a + 0.5 * self.draws) / max(self.games, 1)

    @property
    def smoothed_score(self) -> float:
        """Score with :data:`PRIOR_GAMES` virtual draws mixed in.

        A raw shutout implies an infinite Elo gap; the prior keeps the estimate
        finite and honest about how little a 40-0 result actually pins down.
        """
        n = self.games + PRIOR_GAMES
        return (self.wins_a + 0.5 * self.draws + 0.5 * PRIOR_GAMES) / n

    @property
    def elo_diff(self) -> float:
        return score_to_elo(self.smoothed_score)

    @property
    def elo_ci95(self) -> float:
        """Half-width of a 95% interval on the Elo difference."""
        n = self.games + PRIOR_GAMES
        s = self.smoothed_score
        se = math.sqrt(max(s * (1 - s), 1e-9) / n)
        # d(elo)/d(score) for the logistic model.
        slope = 400.0 / (math.log(10) * s * (1 - s))
        return 1.96 * se * slope

    def __str__(self) -> str:
        return (
            f"{self.name_a} vs {self.name_b}: "
            f"+{self.wins_a} -{self.wins_b} ={self.draws} "
            f"({100 * self.score_a:.1f}%)  "
            f"Elo {self.elo_diff:+.0f} +/- {self.elo_ci95:.0f}  "
            f"[{self.mean_plies:.0f} plies avg]"
        )


def decide_by_progress(st: np.ndarray, scratch: np.ndarray) -> int:
    """Adjudicate an unfinished game by remaining distance to goal.

    Quoridor has no repetition rule, so a player who is losing can simply refuse
    to advance and shuffle forever. Left alone this produces a large fraction of
    ply-capped games, which burn self-play compute and yield value targets of 0
    that teach nothing. Awarding the game to whoever is closer to home resolves
    them and rewards making progress, which is the right incentive anyway.
    """
    d0 = fr.shortest_path_len(st, 0, scratch)
    d1 = fr.shortest_path_len(st, 1, scratch)
    if d0 < d1:
        return 0
    if d1 < d0:
        return 1
    return DRAW


def score_to_elo(score: float) -> float:
    """Elo difference implied by a score rate, clamped for shutout results."""
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def play_game(agent0, agent1, max_plies: int = MAX_PLIES) -> tuple[int, int]:
    """Play one game; ``agent0`` moves first. Returns ``(winner, plies)``.

    ``winner`` is 0, 1, or :data:`DRAW` if the ply cap was hit.
    """
    st = fr.initial_state()
    scratch = fr.make_scratch()
    agents = (agent0, agent1)
    for a in agents:
        if hasattr(a, "reset"):
            a.reset()
    for ply in range(max_plies):
        w = fr.winner(st)
        if w >= 0:
            return w, ply
        action = agents[int(st[fr.IDX_TURN])].select_action(st)
        fr.apply_action(st, int(action), scratch)
    w = fr.winner(st)
    return (w if w >= 0 else DRAW), max_plies


def _play_one(args) -> tuple[int, int]:
    """Worker entry point: build fresh agents, play one game at a given colour."""
    factory_a, factory_b, a_is_first, max_plies = args
    agent_a = factory_a()
    agent_b = factory_b()
    if a_is_first:
        winner, plies = play_game(agent_a, agent_b, max_plies)
        result = 0 if winner == 0 else (1 if winner == 1 else DRAW)
    else:
        winner, plies = play_game(agent_b, agent_a, max_plies)
        result = 1 if winner == 0 else (0 if winner == 1 else DRAW)
    return result, plies


def match(
    factory_a: AgentFactory,
    factory_b: AgentFactory,
    games: int = 100,
    name_a: str = "A",
    name_b: str = "B",
    workers: int = 1,
    max_plies: int = MAX_PLIES,
    progress: bool = False,
) -> MatchResult:
    """Play ``games`` games with alternating colours and summarise the result."""
    jobs = [(factory_a, factory_b, i % 2 == 0, max_plies) for i in range(games)]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_play_one, jobs, chunksize=1))
    else:
        it = jobs
        if progress:
            from tqdm import tqdm

            it = tqdm(jobs, desc=f"{name_a} vs {name_b}", leave=False)
        results = [_play_one(j) for j in it]

    wins_a = sum(1 for r, _ in results if r == 0)
    wins_b = sum(1 for r, _ in results if r == 1)
    draws = sum(1 for r, _ in results if r == DRAW)
    mean_plies = float(np.mean([p for _, p in results])) if results else 0.0
    return MatchResult(name_a, name_b, wins_a, wins_b, draws, mean_plies)


def round_robin(
    factories: Sequence[tuple[str, AgentFactory]],
    games_per_pair: int = 60,
    workers: int = 1,
    max_plies: int = MAX_PLIES,
) -> tuple[list[MatchResult], dict[str, float]]:
    """Play every pairing and fit Elo ratings to the results.

    Returns the individual match results plus a rating table anchored so the
    first listed agent sits at 0.
    """
    results: list[MatchResult] = []
    for i in range(len(factories)):
        for j in range(i + 1, len(factories)):
            name_a, fa = factories[i]
            name_b, fb = factories[j]
            results.append(
                match(fa, fb, games_per_pair, name_a, name_b, workers, max_plies)
            )
    ratings = fit_elo([(r.name_a, r.name_b, r.score_a, r.games) for r in results],
                      [n for n, _ in factories])
    return results, ratings


def fit_elo(
    pairwise: Sequence[tuple[str, str, float, int]],
    names: Sequence[str],
    iters: int = 4000,
    lr: float = 6.0,
    prior_games: float = PRIOR_GAMES,
) -> dict[str, float]:
    """Maximum-likelihood Elo fit over pairwise scores (simple gradient ascent).

    ``pairwise`` entries are ``(name_a, name_b, score_a, games)``. Ratings are
    anchored so ``names[0]`` is 0, since only differences are identifiable.

    Virtual draws are folded into every pairing first. Without them a ladder of
    shutouts is a complete-separation problem and the likelihood is maximised by
    sending ratings to infinity -- the fit simply never converges.
    """
    obs: list[tuple[str, str, float, float]] = []
    for a, b, score_a, n_games in pairwise:
        n_eff = n_games + prior_games
        s_eff = (score_a * n_games + 0.5 * prior_games) / n_eff
        obs.append((a, b, s_eff, n_eff))

    rating = {n: 0.0 for n in names}
    total = max(sum(n for _, _, _, n in obs), 1.0)
    for _ in range(iters):
        grad = {n: 0.0 for n in names}
        for a, b, score_a, n_games in obs:
            expected = elo_to_score(rating[a] - rating[b])
            err = (score_a - expected) * n_games
            grad[a] += err
            grad[b] -= err
        scale = lr / total
        for n in names:
            rating[n] += scale * grad[n] * 400.0 / math.log(10)
        anchor = rating[names[0]]
        for n in names:
            rating[n] -= anchor
    return rating


def batched_match(
    evaluator_a,
    evaluator_b,
    games: int = 128,
    sims: int = 200,
    name_a: str = "A",
    name_b: str = "B",
    c_puct: float = 1.6,
    fpu_reduction: float = 0.2,
    max_plies: int = MAX_PLIES,
    temp_plies: int = 8,
    temperature: float = 1.0,
    max_nodes: int | None = None,
    seed: int = 0,
    sims_b: int | None = None,
) -> MatchResult:
    """Play ``games`` neural-vs-neural games concurrently, sharing GPU batches.

    Single-game MCTS would run the network at batch size 1 and waste the GPU, so
    every game steps in lockstep: at ply ``t`` the side to move is ``t % 2`` in
    every game, which lets both agents batch their halves of the field.

    Opening moves are sampled rather than taken greedily. Two deterministic agents
    would otherwise play one identical game per colour assignment, and a "128-game
    match" would really be two games reported 64 times each.
    """
    from quoridor.mcts import BatchedMCTS  # local: keeps torch out of CPU workers

    rng = np.random.default_rng(seed)
    # ``sims_b`` lets the two sides search at different depths, e.g. to use a
    # low-sim version of a network as an intermediate ladder rung.
    sims_b = sims if sims_b is None else sims_b
    nodes_a = max_nodes if max_nodes is not None else sims + 64
    nodes_b = max_nodes if max_nodes is not None else sims_b + 64
    mcts_a = BatchedMCTS(evaluator_a, n_games=games, max_nodes=nodes_a,
                         c_puct=c_puct, fpu_reduction=fpu_reduction, seed=seed)
    mcts_b = BatchedMCTS(evaluator_b, n_games=games, max_nodes=nodes_b,
                         c_puct=c_puct, fpu_reduction=fpu_reduction, seed=seed + 1)

    states = np.stack([fr.initial_state() for _ in range(games)])
    scratch = fr.make_scratch()
    a_is_first = np.zeros(games, dtype=bool)
    a_is_first[::2] = True
    active = np.ones(games, dtype=bool)
    winners = np.full(games, DRAW, dtype=np.int32)
    plies = np.zeros(games, dtype=np.int32)

    for ply in range(max_plies):
        if not active.any():
            break
        mover_is_a = a_is_first == (ply % 2 == 0)
        for mcts, side_sims, sel in (
            (mcts_a, sims, active & mover_is_a),
            (mcts_b, sims_b, active & ~mover_is_a),
        ):
            idx = np.nonzero(sel)[0]
            if not len(idx):
                continue
            visits = mcts.search(states[idx], side_sims)
            for k, g in enumerate(idx):
                total = visits[k].sum()
                if total <= 0:
                    continue
                if ply < temp_plies and temperature > 1e-3:
                    p = np.power(visits[k] / total, 1.0 / temperature)
                    p = p / p.sum()
                    action = int(rng.choice(len(p), p=p))
                else:
                    action = int(visits[k].argmax())
                fr.apply_action(states[g], action, scratch)
                plies[g] += 1

        for g in np.nonzero(active)[0]:
            w = fr.winner(states[g])
            if w >= 0:
                winners[g] = w
                active[g] = False

    # Adjudicate anything that ran out of plies rather than calling it a draw.
    for g in np.nonzero(active)[0]:
        winners[g] = decide_by_progress(states[g], scratch)

    wins_a = wins_b = draws = 0
    for g in range(games):
        if winners[g] == DRAW:
            draws += 1
        elif (winners[g] == 0) == bool(a_is_first[g]):
            wins_a += 1
        else:
            wins_b += 1
    return MatchResult(name_a, name_b, wins_a, wins_b, draws, float(plies.mean()))


def match_neural_vs_agent(
    evaluator,
    agent,
    games: int = 64,
    sims: int = 200,
    name_a: str = "net",
    name_b: str = "agent",
    c_puct: float = 1.6,
    fpu_reduction: float = 0.2,
    max_plies: int = MAX_PLIES,
    temp_plies: int = 8,
    temperature: float = 1.0,
    max_nodes: int | None = None,
    seed: int = 0,
) -> MatchResult:
    """Play a batched network against a conventional (CPU) agent.

    Same lockstep scheme as :func:`batched_match`: the network searches its half
    of the field in one GPU batch while the plain agent iterates over its half.
    """
    from quoridor.mcts import BatchedMCTS

    rng = np.random.default_rng(seed)
    nodes = max_nodes if max_nodes is not None else sims + 64
    mcts = BatchedMCTS(evaluator, n_games=games, max_nodes=nodes,
                       c_puct=c_puct, fpu_reduction=fpu_reduction, seed=seed)

    states = np.stack([fr.initial_state() for _ in range(games)])
    scratch = fr.make_scratch()
    net_is_first = np.zeros(games, dtype=bool)
    net_is_first[::2] = True
    active = np.ones(games, dtype=bool)
    winners = np.full(games, DRAW, dtype=np.int32)
    plies = np.zeros(games, dtype=np.int32)

    for ply in range(max_plies):
        if not active.any():
            break
        net_moves = active & (net_is_first == (ply % 2 == 0))

        idx = np.nonzero(net_moves)[0]
        if len(idx):
            visits = mcts.search(states[idx], sims)
            for k, g in enumerate(idx):
                total = visits[k].sum()
                if total <= 0:
                    continue
                if ply < temp_plies and temperature > 1e-3:
                    p = np.power(visits[k] / total, 1.0 / temperature)
                    p = p / p.sum()
                    action = int(rng.choice(len(p), p=p))
                else:
                    action = int(visits[k].argmax())
                fr.apply_action(states[g], action, scratch)
                plies[g] += 1

        for g in np.nonzero(active & ~net_moves)[0]:
            if fr.winner(states[g]) >= 0:
                continue
            fr.apply_action(states[g], int(agent.select_action(states[g])), scratch)
            plies[g] += 1

        for g in np.nonzero(active)[0]:
            w = fr.winner(states[g])
            if w >= 0:
                winners[g] = w
                active[g] = False

    # Adjudicate anything that ran out of plies rather than calling it a draw.
    for g in np.nonzero(active)[0]:
        winners[g] = decide_by_progress(states[g], scratch)

    wins_a = wins_b = draws = 0
    for g in range(games):
        if winners[g] == DRAW:
            draws += 1
        elif (winners[g] == 0) == bool(net_is_first[g]):
            wins_a += 1
        else:
            wins_b += 1
    return MatchResult(name_a, name_b, wins_a, wins_b, draws, float(plies.mean()))


def format_ratings(ratings: dict[str, float]) -> str:
    lines = ["", f"{'agent':<28} {'Elo':>8}", "-" * 38]
    for name, r in sorted(ratings.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name:<28} {r:>8.0f}")
    return "\n".join(lines)
