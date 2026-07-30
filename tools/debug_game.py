"""Play one game between two named agents and print every move.

Diagnostic tool: short or lopsided arena results are usually easiest to explain
by just watching a game.
"""

from __future__ import annotations

import argparse

import numpy as np

from quoridor import fastrules as fr
from quoridor import pyrules as pr
from quoridor.agents import GreedyAgent, HeuristicAgent, RandomAgent

MOVE_NAMES = ["N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW"]


def make(name: str):
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name.startswith("heuristic-d"):
        return HeuristicAgent(depth=int(name.split("d")[1]))
    raise SystemExit(f"unknown agent {name}")


def describe(action: int) -> str:
    if action >= fr.MOVE_BASE:
        return f"move {MOVE_NAMES[action - fr.MOVE_BASE]}"
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, fr.W)
    return f"wall {'H' if orient == 0 else 'V'}({wr},{wc})"


def to_py(st: np.ndarray) -> pr.State:
    s = pr.State()
    s.walls_h = [int(x) for x in st[fr.WH_OFF:fr.WH_OFF + 64]]
    s.walls_v = [int(x) for x in st[fr.WV_OFF:fr.WV_OFF + 64]]
    s.pawns = [int(st[fr.IDX_P0]), int(st[fr.IDX_P1])]
    s.walls_left = [int(st[fr.IDX_WL0]), int(st[fr.IDX_WL1])]
    s.turn = int(st[fr.IDX_TURN])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="greedy", help="agent playing first (A)")
    ap.add_argument("--b", default="heuristic-d2", help="agent playing second (B)")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--boards", action="store_true", help="print the board each ply")
    args = ap.parse_args()

    fr.warmup()
    agents = [make(args.a), make(args.b)]
    st = fr.initial_state()
    scratch = fr.make_scratch()

    print(f"A (player 0) = {args.a}\nB (player 1) = {args.b}\n")
    for ply in range(args.max_plies):
        w = fr.winner(st)
        if w >= 0:
            print(f"\n=> {'A' if w == 0 else 'B'} wins after {ply} plies")
            print(to_py(st).render())
            return
        turn = int(st[fr.IDX_TURN])
        action = int(agents[turn].select_action(st))
        d_me = fr.shortest_path_len(st, turn, scratch)
        d_op = fr.shortest_path_len(st, 1 - turn, scratch)
        print(
            f"ply {ply:>3}  {'A' if turn == 0 else 'B'}  {describe(action):<16}"
            f"  dist(self)={d_me:<3} dist(opp)={d_op:<3}"
            f"  walls={int(st[fr.IDX_WL0 + turn])}"
        )
        fr.apply_action(st, action, scratch)
        if args.boards:
            print(to_py(st).render())
    print("\n=> ply cap reached, no winner")


if __name__ == "__main__":
    main()
