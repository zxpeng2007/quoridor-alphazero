"""Main AlphaZero training loop.

Each iteration: generate self-play games with the current best network, train a
candidate on the replay buffer, then gate promotion on an arena match against the
incumbent. Gating matters -- unconditionally accepting each new network lets a bad
training step poison all subsequent self-play data, and the run can spiral without
any obvious symptom.

Elo is re-measured every iteration against a *growing gauntlet* of past
checkpoints rather than against the fixed heuristic baselines. Fixed baselines
saturate almost immediately -- the very first promoted network already beats the
strongest heuristic 64-0, after which that anchor reports a flat ceiling no matter
how much the engine improves. The gauntlet adds a new reference whenever the
engine outgrows the newest one, so the scale keeps extending.

Run state (network, buffer, Elo history, gauntlet) is checkpointed every
iteration so a run can be stopped and resumed.
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
from quoridor.replay import MixedBuffer, ReplayBuffer
from quoridor.selfplay import SelfPlayConfig, SelfPlayEngine
from quoridor.train import Trainer, TrainConfig

# Add a new gauntlet reference once the engine is this far past the newest one.
NEW_REFERENCE_MARGIN = 250.0
# A match this lopsided pins down almost nothing; prefer a closer reference.
SATURATED = 0.97


def anchored_elo(best_net, references, device, args, seed):
    """Estimate the current network's Elo against known-rating references.

    Returns ``(elo, per_reference_detail)``. References whose match saturates are
    dropped when a non-saturated one exists, since a shutout only establishes a
    lower bound and would drag the estimate toward the prior's clamp.
    """
    ev = NetEvaluator(best_net, device=device)
    results = []
    for i, ref in enumerate(references[-args.elo_refs:]):
        ref_net, _ = load_checkpoint(ref["path"], device)
        r = batched_match(
            ev, NetEvaluator(ref_net, device=device),
            games=args.elo_games, sims=args.eval_sims,
            name_a="best", name_b=ref["name"],
            c_puct=args.c_puct, fpu_reduction=args.fpu, seed=seed * 31 + i,
        )
        results.append((ref, r, ref["elo"] + score_to_elo(r.smoothed_score)))
        del ref_net

    usable = [x for x in results if x[1].score_a < SATURATED] or results
    elo = float(np.mean([x[2] for x in usable]))
    detail = "  ".join(
        f"{ref['name']}@{ref['elo']:.0f}: {100 * r.score_a:.0f}%" for ref, r, _ in results
    )
    return elo, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/bootstrap.pt")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--games-per-iter", type=int, default=4000)
    ap.add_argument("--steps-per-iter", type=int, default=900)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--parallel", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-sims", type=int, default=200)
    ap.add_argument("--promote-threshold", type=float, default=0.55)
    ap.add_argument("--elo-games", type=int, default=48, help="games per gauntlet match")
    ap.add_argument("--elo-refs", type=int, default=2, help="references per iteration")
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mined", default="",
                    help="npz of deep-search-labelled positions from "
                         "tools/mine_disagreements.py, oversampled into training")
    ap.add_argument("--mined-frac", type=float, default=0.25,
                    help="fraction of each batch drawn from the mined set")
    ap.add_argument("--mined-holdout", default="",
                    help="held-out mined npz; value MAE on it is reported every "
                         "iteration, since the 200-sim gates cannot see this repair")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "refs").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()

    best_path = out / "best.pt"
    history_path = out / "history.json"
    refs_path = out / "references.json"
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
        best = QuoridorNet(NetConfig(channels=args.channels, blocks=args.blocks)).to(device)
        print("initialised from scratch (no bootstrap checkpoint found)")
        print("  warning: an unbootstrapped policy makes MCTS nearly blind in a")
        print("  140-action space; tools/bootstrap.py first is strongly advised.")

    # The gauntlet is anchored at the bootstrap network = 0 Elo, so every number
    # reported is "Elo gained over the supervised starting point".
    if refs_path.exists():
        references = json.loads(refs_path.read_text())
    else:
        anchor = out / "refs" / "ref-bootstrap.pt"
        if not anchor.exists():
            src, _ = load_checkpoint(args.init, device)
            save_checkpoint(anchor, src, meta={"reference": True})
        references = [{"name": "bootstrap", "path": str(anchor), "elo": 0.0}]
        refs_path.write_text(json.dumps(references, indent=2))
    print(f"gauntlet: {len(references)} reference(s), newest at "
          f"{references[-1]['elo']:+.0f} Elo")
    print(f"net: {args.blocks}x{args.channels}, {best.num_params() / 1e6:.2f}M params, {device}\n")

    buffer = ReplayBuffer(capacity=args.buffer)
    buf_path = out / "buffer.npz"
    if args.resume and buf_path.exists():
        buffer = ReplayBuffer.load(str(buf_path), capacity=args.buffer)
        print(f"resumed replay buffer: {len(buffer):,} positions\n")

    train_source = buffer
    if args.mined:
        blob = np.load(args.mined)
        repair = ReplayBuffer(capacity=max(len(blob["states"]), 1))
        repair.add(blob["states"], blob["policies"], blob["values"])
        train_source = MixedBuffer(buffer, repair, frac=args.mined_frac)
        print(f"repair set: {len(repair):,} mined positions, "
              f"{100 * args.mined_frac:.0f}% of every batch")

    holdout = None
    if args.mined_holdout:
        hb = np.load(args.mined_holdout)
        holdout = (hb["states"], hb["values"].astype(np.float32))
        gap = np.abs(hb["raw"] - hb["values"]).mean() if "raw" in hb else float("nan")
        print(f"value gate: {len(holdout[0]):,} held-out positions, "
              f"raw-vs-deep MAE at mining time {gap:.3f}\n")

    sp_cfg = SelfPlayConfig(
        n_parallel=args.parallel, sims=args.sims, c_puct=args.c_puct, seed=start_iter
    )
    engine = SelfPlayEngine(NetEvaluator(best, device=device), sp_cfg)

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
        stats = trainer.train_on_buffer(train_source, args.steps_per_iter)
        print(f"  train ({time.perf_counter() - t0:.0f}s): {stats}")

        gate_mae = float("nan")
        if holdout is not None:
            gate_mae = trainer.value_mae_on(holdout[0], holdout[1])
            print(f"  value gate: MAE {gate_mae:.3f} on held-out "
                  f"disagreement positions")

        # 3. gate promotion on a head-to-head match against the incumbent
        t0 = time.perf_counter()
        result = batched_match(
            NetEvaluator(candidate, device=device),
            NetEvaluator(best, device=device),
            games=args.eval_games, sims=args.eval_sims,
            name_a=f"cand-{it}", name_b="best",
            c_puct=args.c_puct, fpu_reduction=args.fpu, seed=it,
        )
        promoted = result.score_a >= args.promote_threshold
        print(f"  arena ({time.perf_counter() - t0:.0f}s): {result}")
        print(f"  -> {'PROMOTED' if promoted else 'rejected'} "
              f"(needs {100 * args.promote_threshold:.0f}%)")
        if promoted:
            best = candidate

        # 4. re-measure Elo against the gauntlet, every iteration
        t0 = time.perf_counter()
        elo, detail = anchored_elo(best, references, device, args, it)
        print(f"  elo ({time.perf_counter() - t0:.0f}s): {elo:+.0f} "
              f"over bootstrap   [{detail}]")

        if elo - references[-1]["elo"] > NEW_REFERENCE_MARGIN:
            ref_file = out / "refs" / f"ref-{it:03d}.pt"
            save_checkpoint(ref_file, best, meta={"reference": True, "elo": elo})
            references.append({"name": f"gen-{it:03d}", "path": str(ref_file), "elo": elo})
            refs_path.write_text(json.dumps(references, indent=2))
            print(f"  + new gauntlet reference at {elo:+.0f} Elo "
                  f"({len(references)} total)")

        # 5. persist
        save_checkpoint(best_path, best, meta={
            "iteration": it, "promoted": promoted,
            "positions": buffer.total_added, "elo": elo,
        })
        if promoted:
            save_checkpoint(out / f"gen-{it:03d}.pt", best,
                            meta={"iteration": it, "elo": elo})

        history.append({
            "iteration": it,
            "games": buffer.total_added,
            "promoted": bool(promoted),
            "score_vs_best": result.score_a,
            "elo_gain": score_to_elo(result.smoothed_score),
            "elo": elo,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "policy_top1": stats.policy_acc,
            "value_mae": stats.value_mae,
            "gate_value_mae": gate_mae,
            "mean_plies": batch.stats.mean_plies,
            "p0_win_rate": batch.stats.p0_win_rate,
            "seconds": time.perf_counter() - t_iter,
        })
        history_path.write_text(json.dumps(history, indent=2))
        if it % 5 == 0:
            buffer.save(str(buf_path))

        print(f"  iteration took {time.perf_counter() - t_iter:.0f}s | "
              f"Elo {elo:+.0f}\n")


if __name__ == "__main__":
    main()
