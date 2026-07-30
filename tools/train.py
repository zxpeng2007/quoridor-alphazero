"""Main AlphaZero training loop.

Each iteration: generate self-play games with the current best network, train a
candidate on the replay buffer, then gate promotion on an arena match against the
incumbent. Gating matters -- unconditionally accepting each new network lets a bad
training step poison all subsequent self-play data, and the run can spiral without
any obvious symptom.

Run state (network, buffer, Elo history) is checkpointed every iteration so a run
can be stopped and resumed.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.arena import batched_match, score_to_elo
from quoridor.net import NetConfig, NetEvaluator, QuoridorNet, load_checkpoint, save_checkpoint
from quoridor.replay import ReplayBuffer
from quoridor.selfplay import SelfPlayConfig, SelfPlayEngine
from quoridor.train import Trainer, TrainConfig


def build_net(args) -> QuoridorNet:
    return QuoridorNet(NetConfig(channels=args.channels, blocks=args.blocks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/bootstrap.pt",
                    help="starting checkpoint (from tools/bootstrap.py)")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--games-per-iter", type=int, default=3000)
    ap.add_argument("--steps-per-iter", type=int, default=800)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--promote-threshold", type=float, default=0.55)
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()

    # ------------------------------------------------------------- load state
    best_path = out / "best.pt"
    history_path = out / "history.json"
    history: list[dict] = []
    start_iter = 0

    if args.resume and best_path.exists():
        best, meta = load_checkpoint(best_path, device)
        start_iter = int(meta.get("iteration", 0)) + 1
        if history_path.exists():
            history = json.loads(history_path.read_text())
        print(f"resumed from {best_path} at iteration {start_iter}")
    elif Path(args.init).exists():
        best, _ = load_checkpoint(args.init, device)
        print(f"initialised from {args.init}")
    else:
        best = build_net(args).to(device)
        print("initialised from scratch (no bootstrap checkpoint found)")
        print("  warning: an unbootstrapped policy makes MCTS nearly blind in a")
        print("  140-action space; tools/bootstrap.py first is strongly advised.")

    print(f"net: {args.blocks}x{args.channels}, {best.num_params() / 1e6:.2f}M params, {device}\n")

    buffer = ReplayBuffer(capacity=args.buffer)
    buf_path = out / "buffer.npz"
    if args.resume and buf_path.exists():
        buffer = ReplayBuffer.load(str(buf_path), capacity=args.buffer)
        print(f"resumed replay buffer: {len(buffer):,} positions\n")

    sp_cfg = SelfPlayConfig(
        n_parallel=args.parallel, sims=args.sims, c_puct=args.c_puct, seed=start_iter
    )
    # The search tree arrays are hundreds of MB; build the engine once and just
    # point it at the current network each iteration.
    engine = SelfPlayEngine(NetEvaluator(best, device=device), sp_cfg)

    # ------------------------------------------------------------------ loop
    for it in range(start_iter, args.iterations):
        t_iter = time.perf_counter()
        print(f"=== iteration {it} " + "=" * 46)

        # 1. self-play with the incumbent
        t0 = time.perf_counter()
        engine.mcts.evaluator = NetEvaluator(best, device=device)
        batch = engine.generate(args.games_per_iter, progress=True)
        buffer.add(batch.states, batch.policies, batch.values)
        t_sp = time.perf_counter() - t0
        print(f"  self-play: {batch.stats}")
        print(f"    {t_sp:.0f}s ({args.games_per_iter / t_sp * 3600:,.0f} games/hr), "
              f"buffer {len(buffer):,}")

        # 2. train a candidate from a copy of the incumbent
        candidate = copy.deepcopy(best)
        trainer = Trainer(
            candidate,
            TrainConfig(batch_size=args.batch_size, lr=args.lr, warmup_steps=50),
            device=device,
        )
        t0 = time.perf_counter()
        stats = trainer.train_on_buffer(buffer, args.steps_per_iter)
        print(f"  train ({time.perf_counter() - t0:.0f}s): {stats}")

        # 3. gate promotion on a head-to-head match
        t0 = time.perf_counter()
        result = batched_match(
            NetEvaluator(candidate, device=device),
            NetEvaluator(best, device=device),
            games=args.eval_games,
            sims=args.eval_sims,
            name_a=f"cand-{it}",
            name_b="best",
            c_puct=args.c_puct,
            fpu_reduction=args.fpu,
            seed=it,
        )
        promoted = result.score_a >= args.promote_threshold
        print(f"  arena ({time.perf_counter() - t0:.0f}s): {result}")
        print(f"  -> {'PROMOTED' if promoted else 'rejected'} "
              f"(needs {100 * args.promote_threshold:.0f}%)")

        if promoted:
            best = candidate

        # 4. persist
        save_checkpoint(
            best_path, best,
            meta={
                "iteration": it,
                "promoted": promoted,
                "positions": buffer.total_added,
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
            },
        )
        if promoted:
            save_checkpoint(out / f"gen-{it:03d}.pt", best, meta={"iteration": it})

        history.append({
            "iteration": it,
            "games": buffer.total_added,
            "promoted": bool(promoted),
            "score_vs_best": result.score_a,
            "elo_gain": score_to_elo(result.smoothed_score),
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "policy_top1": stats.policy_acc,
            "value_mae": stats.value_mae,
            "mean_plies": batch.stats.mean_plies,
            "p0_win_rate": batch.stats.p0_win_rate,
            "seconds": time.perf_counter() - t_iter,
        })
        history_path.write_text(json.dumps(history, indent=2))
        if it % 5 == 0:
            buffer.save(str(buf_path))

        cum = sum(h["elo_gain"] for h in history if h["promoted"])
        print(f"  iteration took {time.perf_counter() - t_iter:.0f}s | "
              f"cumulative Elo gain ~{cum:+.0f}\n")


if __name__ == "__main__":
    main()
