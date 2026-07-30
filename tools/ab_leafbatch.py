"""Equal-time A/B: leaf-batched search vs the current single-leaf search.

Both sides get the same wall-clock budget per move (measured, not assumed).
If the batched side wins clearly, the virtual-loss distortion costs less than
the extra depth buys, and it becomes the default for interactive play.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.arena import MatchResult, decide_by_progress
from quoridor.mcts import BatchedMCTS, visits_to_policy
from quoridor.net import NetEvaluator, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--sims-a", type=int, default=4096, help="batched side")
    ap.add_argument("--leaf-a", type=int, default=32)
    ap.add_argument("--sims-b", type=int, default=600, help="single-leaf side")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()
    net, _ = load_checkpoint(args.checkpoint, device)
    ev = NetEvaluator(net, device=device, graph_batches=(1, args.leaf_a))

    mcts_a = BatchedMCTS(ev, n_games=1, max_nodes=args.sims_a + 128,
                         leaf_batch=args.leaf_a, seed=1)
    mcts_b = BatchedMCTS(ev, n_games=1, max_nodes=args.sims_b + 128, seed=2)
    rng = np.random.default_rng(7)
    scratch = fr.make_scratch()

    # Measure actual per-move cost so the time claim is checked, not asserted.
    st0 = fr.initial_state().reshape(1, -1)
    for m, sims in ((mcts_a, args.sims_a), (mcts_b, args.sims_b)):
        m.search(st0.copy(), sims)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mcts_a.search(st0.copy(), args.sims_a)
    torch.cuda.synchronize()
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    mcts_b.search(st0.copy(), args.sims_b)
    torch.cuda.synchronize()
    tb = time.perf_counter() - t0
    print(f"A: {args.sims_a} sims @ leaf_batch={args.leaf_a}: {ta * 1e3:.0f} ms/move")
    print(f"B: {args.sims_b} sims @ leaf_batch=1:  {tb * 1e3:.0f} ms/move")
    print(f"time ratio A/B: {ta / tb:.2f} (should be ~1 for a fair fight)\n")

    wins = [0, 0]
    draws = 0
    plies_total = 0
    t_start = time.perf_counter()
    for g in range(args.games):
        a_first = g % 2 == 0
        st = fr.initial_state()
        for ply in range(300):
            w = fr.winner(st)
            if w >= 0:
                break
            a_moves = (int(st[fr.IDX_TURN]) == 0) == a_first
            mcts, sims = (mcts_a, args.sims_a) if a_moves else (mcts_b, args.sims_b)
            visits = mcts.search(st.reshape(1, -1).copy(), sims)
            if ply < 8:
                p = visits_to_policy(visits, 1.0)[0]
                action = int(rng.choice(len(p), p=p))
            else:
                action = int(visits[0].argmax())
            fr.apply_action(st, action, scratch)
        w = fr.winner(st)
        if w < 0:
            w = decide_by_progress(st, scratch)
        plies_total += ply
        if w < 0:
            draws += 1
        else:
            winner_is_a = (w == 0) == a_first
            wins[0 if winner_is_a else 1] += 1
        if (g + 1) % 8 == 0:
            print(f"  {g + 1}/{args.games}: A +{wins[0]} -{wins[1]} ={draws}",
                  flush=True)

    r = MatchResult(f"L{args.leaf_a}@{args.sims_a}", f"L1@{args.sims_b}",
                    wins[0], wins[1], draws, plies_total / max(args.games, 1))
    print(f"\n{r}")
    print(f"total: {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
