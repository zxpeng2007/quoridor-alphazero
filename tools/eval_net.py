"""Evaluate a trained checkpoint against the baseline ladder.

Anchors the network's strength to the same scale produced by
``tools/eval_baselines.py`` (random = 0), so progress across a training run is
comparable rather than only relative to the previous checkpoint.
"""

from __future__ import annotations

import argparse
import time

import torch

from quoridor import fastrules as fr
from quoridor.agents import GreedyAgent, HeuristicAgent, RandomAgent
from quoridor.arena import batched_match, match_neural_vs_agent, score_to_elo
from quoridor.net import NetEvaluator, load_checkpoint

# Elo of each baseline on the random=0 scale, from tools/eval_baselines.py.
BASELINE_ELO = {
    "random": 0.0,
    "greedy": 527.0,
    "heuristic-d1": 527.0,
    "heuristic-d2": 995.0,
    "heuristic-d3": 1412.0,
}


def make_baseline(name: str):
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    return HeuristicAgent(depth=int(name.split("d")[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--vs", nargs="*", default=["greedy", "heuristic-d2", "heuristic-d3"])
    ap.add_argument("--opponent", default="", help="also play against this checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()

    net, meta = load_checkpoint(args.checkpoint, device)
    ev = NetEvaluator(net, device=device)
    print(f"checkpoint: {args.checkpoint}")
    if meta:
        print(f"  meta: {meta}")
    print(f"  {args.sims} sims/move, {args.games} games per opponent\n")

    estimates = []
    for name in args.vs:
        t0 = time.perf_counter()
        result = match_neural_vs_agent(
            ev, make_baseline(name), games=args.games, sims=args.sims,
            name_a="net", name_b=name, c_puct=args.c_puct, fpu_reduction=args.fpu,
        )
        print(f"  {result}   [{time.perf_counter() - t0:.0f}s]")
        if name in BASELINE_ELO:
            estimates.append(BASELINE_ELO[name] + score_to_elo(result.smoothed_score))

    if args.opponent:
        other, _ = load_checkpoint(args.opponent, device)
        result = batched_match(
            ev, NetEvaluator(other, device=device),
            games=args.games, sims=args.sims,
            name_a="net", name_b=args.opponent,
            c_puct=args.c_puct, fpu_reduction=args.fpu,
        )
        print(f"  {result}")

    if estimates:
        print(f"\nestimated Elo (random=0 scale): {sum(estimates) / len(estimates):.0f}")
        print(f"  per-opponent estimates: {[f'{e:.0f}' for e in estimates]}")
        print("  (wide spread across opponents means the anchors disagree and the"
              "\n   single number should not be trusted)")


if __name__ == "__main__":
    main()
