"""Full-ladder strength evaluation: one joint Elo fit over the whole run.

Measures the trained engine's strength properly, rather than trusting the
incremental gauntlet numbers from training. The design constraint is that a
shutout match is a complete-separation case and pins down almost nothing, so:

* the neural generations are chained through *adjacent* matches (which training
  showed land in the informative 55-90% band),
* the classical agents are bridged to the neural ladder via *weakened* versions
  of the bootstrap network (16/64 sims) rather than full-strength shutouts,
* any match that still saturates (>=97% either way) is printed as a
  verification line but EXCLUDED from the joint fit, so a clamped shutout
  cannot drag the maximum-likelihood ratings around.

Ratings are anchored at bootstrap@200 = 0, the same anchor the training
gauntlet uses. These are internal, self-consistent numbers: they say nothing
directly about any external ladder (e.g. barricade.gg's own ratings).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch

from quoridor import fastrules as fr
from quoridor.agents import HeuristicAgent
from quoridor.arena import (
    MatchResult,
    batched_match,
    fit_elo,
    match_neural_vs_agent,
)
from quoridor.net import NetEvaluator, load_checkpoint

SATURATED = 0.97  # matches outside [1-SAT, SAT] are excluded from the fit


class Rungs:
    """Lazy checkpoint loader; keeps evaluators cached (they are small)."""

    def __init__(self, ckpt_dir: Path, device: str):
        self.dir = ckpt_dir
        self.device = device
        self._cache: dict[str, NetEvaluator] = {}

    def evaluator(self, name: str) -> NetEvaluator:
        if name not in self._cache:
            path = self.dir / ("bootstrap.pt" if name == "bootstrap" else f"{name}.pt")
            net, _ = load_checkpoint(path, self.device)
            self._cache[name] = NetEvaluator(net, device=self.device)
        return self._cache[name]


def pick_generations(ckpt_dir: Path, max_rungs: int = 6) -> list[str]:
    """Evenly spaced promoted generations, always including first and latest."""
    gens = sorted(
        (int(m.group(1)), p.stem)
        for p in ckpt_dir.glob("gen-*.pt")
        if (m := re.match(r"gen-(\d+)$", p.stem))
    )
    if not gens:
        return []
    if len(gens) <= max_rungs:
        return [name for _, name in gens]
    idx = [round(i * (len(gens) - 1) / (max_rungs - 1)) for i in range(max_rungs)]
    return [gens[i][1] for i in sorted(set(idx))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="checkpoints")
    ap.add_argument("--games", type=int, default=128, help="games per neural match")
    ap.add_argument("--agent-games", type=int, default=64, help="games vs CPU agents")
    ap.add_argument("--sims", type=int, default=200, help="chain search depth")
    ap.add_argument("--quick", action="store_true", help="half the games everywhere")
    args = ap.parse_args()
    if args.quick:
        args.games //= 2
        args.agent_games //= 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()

    ckpt_dir = Path(args.dir)
    rungs = Rungs(ckpt_dir, device)
    gens = pick_generations(ckpt_dir)
    if not gens:
        raise SystemExit(f"no gen-*.pt checkpoints in {ckpt_dir}")
    latest = gens[-1]
    chain = ["bootstrap"] + gens
    print(f"chain rungs @ {args.sims} sims: {' -> '.join(chain)}")
    print(f"device: {device}\n")

    results: list[MatchResult] = []

    def saturated(r: MatchResult) -> bool:
        return not (1 - SATURATED <= r.score_a <= SATURATED)

    def play_net(name_a, name_b, sims_a, sims_b, games, seed) -> MatchResult:
        t0 = time.perf_counter()
        r = batched_match(
            rungs.evaluator(name_a.split("@")[0]),
            rungs.evaluator(name_b.split("@")[0]),
            games=games, sims=sims_a, sims_b=sims_b,
            name_a=name_a, name_b=name_b, seed=seed,
        )
        results.append(r)
        excl = "   [excluded from fit: saturated]" if saturated(r) else ""
        print(f"  {r}{excl}   [{time.perf_counter() - t0:.0f}s]", flush=True)
        return r

    def play_agent(name_a, sims_a, agent, agent_name, games, seed) -> MatchResult:
        t0 = time.perf_counter()
        r = match_neural_vs_agent(
            rungs.evaluator(name_a.split("@")[0]), agent,
            games=games, sims=sims_a, name_a=name_a, name_b=agent_name, seed=seed,
        )
        results.append(r)
        excl = "   [excluded from fit: saturated]" if saturated(r) else ""
        print(f"  {r}{excl}   [{time.perf_counter() - t0:.0f}s]", flush=True)
        return r

    # ---- 1. the generation chain (adjacent pairs: the informative band)
    #
    # A hop whose direct match saturates would disconnect the chain (the very
    # first hop, bootstrap -> gen-000, is a ~96% matchup and does exactly
    # that). When it happens, bridge through a sims-weakened copy of the newer
    # net: new@64 vs old, plus new vs new@64, both of which land back in the
    # measurable band. One further fallback at @16 covers extreme gaps.
    print("generation chain:")
    for i in range(len(chain) - 1):
        old, new = chain[i], chain[i + 1]
        r = play_net(new, old, args.sims, args.sims, args.games, seed=10 + 3 * i)
        if saturated(r):
            half = f"{new}@64"
            r2 = play_net(half, old, 64, args.sims, args.games, seed=11 + 3 * i)
            play_net(new, half, args.sims, 64, args.games, seed=12 + 3 * i)
            if saturated(r2):
                quarter = f"{new}@16"
                play_net(quarter, old, 16, args.sims, args.games, seed=200 + 3 * i)
                play_net(half, quarter, 64, 16, args.games, seed=201 + 3 * i)

    # ---- 2. sims ladder on the bootstrap net (bridges down to the classics)
    print("bootstrap sims ladder:")
    play_net("bootstrap", "bootstrap@64", args.sims, 64, args.games, seed=40)
    play_net("bootstrap@64", "bootstrap@16", 64, 16, args.games, seed=41)

    # ---- 3. anchors to the classical agents via the weakened net
    print("classical anchors:")
    play_agent("bootstrap@16", 16, HeuristicAgent(depth=2), "heuristic-d2",
               args.agent_games, seed=50)
    play_agent("bootstrap@16", 16, HeuristicAgent(depth=3), "heuristic-d3",
               args.agent_games, seed=51)
    play_agent("bootstrap@64", 64, HeuristicAgent(depth=3), "heuristic-d3",
               args.agent_games, seed=52)

    # ---- 4. cross-checks (expected shutouts; verification only)
    print("cross-checks:")
    play_agent(latest, args.sims, HeuristicAgent(depth=3), "heuristic-d3",
               args.agent_games, seed=60)
    play_net(f"{latest}@800", latest, 800, args.sims, args.games, seed=61)

    # ---- joint fit over the informative matches, anchored bootstrap = 0
    pairwise = []
    names: list[str] = ["bootstrap"]
    for r in results:
        if not (1 - SATURATED <= r.score_a <= SATURATED):
            continue
        pairwise.append((r.name_a, r.name_b, r.score_a, r.games))
        for n in (r.name_a, r.name_b):
            if n not in names:
                names.append(n)

    # Drop any rung that lost its last informative connection to the anchor.
    connected = {"bootstrap"}
    changed = True
    while changed:
        changed = False
        for a, b, _, _ in pairwise:
            if (a in connected) != (b in connected):
                connected |= {a, b}
                changed = True
    fit_pairs = [p for p in pairwise if p[0] in connected and p[1] in connected]
    fit_names = [n for n in names if n in connected]

    ratings = fit_elo(fit_pairs, fit_names)
    print(f"\n{'rung':<24} {'Elo (bootstrap=0)':>18}")
    print("-" * 44)
    for name, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f"{name:<24} {elo:>+18.0f}")
    dropped = [n for n in names if n not in connected]
    if dropped:
        print(f"(unrated, no informative link: {', '.join(dropped)})")

    print(
        "\nNotes:\n"
        "  * Internal, self-consistent scale anchored at the supervised bootstrap\n"
        "    network (200 sims) = 0. Not comparable to any external ladder.\n"
        "  * Saturated matches (>=97%) are shown above but excluded from the fit;\n"
        "    they are lower bounds, not measurements.\n"
    )


if __name__ == "__main__":
    main()
