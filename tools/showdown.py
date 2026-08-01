"""Play a single deep game between two checkpoints and print it move by move.

Both sides get the same, large simulation budget, so the result reflects the
networks rather than a difference in search effort.

    python tools/showdown.py checkpoints/gen-016.pt checkpoints/gen-010.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor import pyrules as pr
from quoridor.arena import decide_by_progress
from quoridor.mcts import BatchedMCTS
from quoridor.net import NetEvaluator, load_checkpoint

FILES = "abcdefghi"
MOVE_NAMES = ["N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW"]


def describe(action: int, st_after: np.ndarray, mover: int) -> str:
    if action >= fr.MOVE_BASE:
        cell = int(st_after[fr.IDX_P0 if mover == 0 else fr.IDX_P1])
        r, c = divmod(cell, 9)
        return f"{FILES[c]}{r + 1}"
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, 8)
    return f"{'h' if orient == 0 else 'v'}{FILES[wc]}{wr + 1}"


def render(st: np.ndarray) -> str:
    s = pr.State()
    s.walls_h = [int(x) for x in st[fr.WH_OFF:fr.WH_OFF + 64]]
    s.walls_v = [int(x) for x in st[fr.WV_OFF:fr.WV_OFF + 64]]
    s.pawns = [int(st[fr.IDX_P0]), int(st[fr.IDX_P1])]
    s.walls_left = [int(st[fr.IDX_WL0]), int(st[fr.IDX_WL1])]
    s.turn = int(st[fr.IDX_TURN])
    return s.render()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("white", help="checkpoint moving first")
    ap.add_argument("black", help="checkpoint moving second")
    ap.add_argument("--sims", type=int, default=400_000)
    ap.add_argument("--leaf-batch", type=int, default=32)
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--max-plies", type=int, default=250)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()

    engines, names = {}, {}
    for seat, path in ((0, args.white), (1, args.black)):
        net, meta = load_checkpoint(path, device)
        ev = NetEvaluator(net, device=device, graph_batches=(args.leaf_batch,))
        engines[seat] = BatchedMCTS(
            ev, n_games=1, max_nodes=args.sims + args.leaf_batch + 256,
            leaf_batch=args.leaf_batch, c_puct=args.c_puct,
            fpu_reduction=args.fpu)
        names[seat] = (f"{path.split('/')[-1].replace('.pt','')} "
                       f"(gen {meta.get('iteration','?')}, "
                       f"{meta.get('elo', 0):+.0f} Elo)")

    print(f"player 1 (first): {names[0]}")
    print(f"player 2        : {names[1]}")
    print(f"{args.sims:,} simulations per move for both sides\n")

    st = fr.initial_state()
    scratch = fr.make_scratch()
    t_start = time.perf_counter()
    for ply in range(args.max_plies):
        w = fr.winner(st)
        if w >= 0:
            break
        mover = int(st[fr.IDX_TURN])
        t0 = time.perf_counter()
        visits = engines[mover].search(st.reshape(1, -1).copy(), args.sims)[0]
        if visits.sum() <= 0:
            break
        value = float(engines[mover].root_value()[0])
        action = int(visits.argmax())
        share = float(visits[action] / visits.sum())
        fr.apply_action(st, action, scratch)
        d0 = fr.shortest_path_len(st, 0, scratch)
        d1 = fr.shortest_path_len(st, 1, scratch)
        print(f"{ply + 1:>3}. P{mover + 1} {describe(action, st, mover):<5} "
              f"eval {value:+.2f}  conviction {100 * share:>3.0f}%  "
              f"dist {d0}v{d1}  walls {int(st[fr.IDX_WL0])}/{int(st[fr.IDX_WL1])}"
              f"  [{time.perf_counter() - t0:.1f}s]", flush=True)

    w = fr.winner(st)
    adjudicated = False
    if w < 0:
        w = decide_by_progress(st, scratch)
        adjudicated = True
    print("\n" + render(st))
    if w < 0:
        print("\nresult: draw")
    else:
        print(f"\nresult: {names[w]} wins"
              + (" (adjudicated on remaining distance)" if adjudicated else ""))
    print(f"game took {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
