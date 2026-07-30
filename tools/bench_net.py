"""Benchmark network inference and end-to-end MCTS throughput.

Sizing the training run depends almost entirely on these numbers: self-play cost
is (games) x (moves/game) x (sims/move) network evaluations, and the 9x9 board is
small enough that GPU efficiency is dominated by batch size, not FLOPs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS
from quoridor.net import NetConfig, NetEvaluator, QuoridorNet


def bench_forward(net: QuoridorNet, device: str) -> None:
    ev = NetEvaluator(net, device=device)
    print(f"{'batch':>8} {'ms/call':>10} {'pos/s':>12}")
    print("-" * 32)
    for batch in (64, 256, 1024, 2048, 4096):
        planes = np.random.rand(batch, 8, 9, 9).astype(np.float32)
        legal = np.ones((batch, fr.NUM_ACTIONS), dtype=np.uint8)
        for _ in range(3):
            ev.evaluate(planes, legal)
        torch.cuda.synchronize()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            ev.evaluate(planes, legal)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        print(f"{batch:>8} {dt * 1e3:>10.2f} {batch / dt:>12,.0f}")
    print()


def bench_selfplay(net: QuoridorNet, device: str, sims: int, games: int) -> None:
    """Measure a full move-round: search all games, then advance each by one ply."""
    ev = NetEvaluator(net, device=device)
    mcts = BatchedMCTS(ev, n_games=games, max_nodes=sims + 64)
    states = np.stack([fr.initial_state() for _ in range(games)])
    scratch = fr.make_scratch()

    mcts.search(states.copy(), n_sims=8)  # warm JIT + cudnn autotune
    torch.cuda.synchronize()

    rounds = 5
    t0 = time.perf_counter()
    for _ in range(rounds):
        visits = mcts.search(states, n_sims=sims)
        for g in range(games):
            if fr.winner(states[g]) < 0 and visits[g].sum() > 0:
                fr.apply_action(states[g], int(visits[g].argmax()), scratch)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rounds

    evals = games * sims
    moves_per_s = games / dt
    print(
        f"games={games:>4} sims={sims:>4}  "
        f"{dt * 1e3:>8.0f} ms/round  "
        f"{evals / dt:>10,.0f} evals/s  "
        f"{moves_per_s:>8,.0f} moves/s"
    )
    return moves_per_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--sims", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    net = QuoridorNet(NetConfig(channels=args.channels, blocks=args.blocks))
    print(f"net: {args.blocks} blocks x {args.channels} ch, "
          f"{net.num_params() / 1e6:.2f}M params, device={device}\n")

    bench_forward(net, device)

    print("end-to-end self-play (search + advance one ply):")
    rates = []
    for games in (64, 128, 256, 512):
        rates.append((games, bench_selfplay(net, device, args.sims, games)))

    best_games, best_rate = max(rates, key=lambda kv: kv[1])
    plies = 70  # typical decisive Quoridor game length
    games_per_hour = best_rate / plies * 3600
    print(
        f"\nbest: {best_games} parallel games -> {best_rate:,.0f} moves/s"
        f"\nat ~{plies} plies/game: {games_per_hour:,.0f} games/hour"
    )


if __name__ == "__main__":
    main()
