"""Round-robin the non-neural baselines to establish the bottom of the ladder.

These numbers are the yardstick the first trained networks get graded against.
"""

from __future__ import annotations

import argparse
import time
from functools import partial

from quoridor import fastrules as fr
from quoridor.agents import GreedyAgent, HeuristicAgent, RandomAgent
from quoridor.arena import format_ratings, round_robin


def timing_probe() -> None:
    """Report per-move cost for each baseline, to size arena runs sensibly."""
    st = fr.initial_state()
    scratch = fr.make_scratch()
    # Advance a few plies so the position is representative, not the empty board.
    for a in (129, 133, 130, 134, 0, 64):
        fr.apply_action(st, a, scratch)

    print(f"{'agent':<20} {'ms/move':>10} {'nodes':>10}")
    print("-" * 42)
    for name, agent in [
        ("random", RandomAgent()),
        ("greedy", GreedyAgent()),
        ("heuristic-d1", HeuristicAgent(depth=1)),
        ("heuristic-d2", HeuristicAgent(depth=2)),
        ("heuristic-d3", HeuristicAgent(depth=3)),
    ]:
        agent.select_action(st)  # warm the JIT
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            agent.select_action(st)
        dt = (time.perf_counter() - t0) / reps
        nodes = getattr(agent, "nodes", 0) // max(reps, 1)
        print(f"{name:<20} {dt * 1e3:>10.2f} {nodes:>10}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60, help="games per pairing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="skip the deepest agent")
    args = ap.parse_args()

    fr.warmup()
    timing_probe()

    factories = [
        ("random", partial(RandomAgent, seed=0)),
        ("greedy", partial(GreedyAgent, seed=0)),
        ("heuristic-d1", partial(HeuristicAgent, depth=1)),
        ("heuristic-d2", partial(HeuristicAgent, depth=2)),
    ]
    if not args.quick:
        factories.append(("heuristic-d3", partial(HeuristicAgent, depth=3)))

    print(f"round-robin: {args.games} games per pairing, {args.workers} workers\n")
    t0 = time.perf_counter()
    results, ratings = round_robin(factories, args.games, workers=args.workers)
    dt = time.perf_counter() - t0

    for r in results:
        print("  " + str(r))
    print(format_ratings(ratings))
    print(f"\ntotal: {dt:.1f}s")


if __name__ == "__main__":
    main()
