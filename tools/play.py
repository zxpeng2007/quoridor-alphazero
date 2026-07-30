"""Play against the engine, or have it analyse a position.

Two modes:

* ``play``    -- interactive game against the engine at a chosen strength
* ``analyse`` -- show the engine's evaluation and top moves for a position

Move notation
-------------
Pawn moves are compass directions: ``N``, ``S``, ``E``, ``W`` for single steps,
``NN``/``SS``/``EE``/``WW`` for jumps over the opponent, and ``NE``/``NW``/``SE``/
``SW`` for the diagonal side-steps available when a straight jump is blocked.

Walls are ``H r c`` or ``V r c`` where ``(r, c)`` is the wall slot, both in 0..7.
A horizontal wall at ``(r, c)`` sits between rows ``r`` and ``r+1`` spanning
columns ``c`` and ``c+1``; a vertical wall at ``(r, c)`` sits between columns
``c`` and ``c+1`` spanning rows ``r`` and ``r+1``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor import pyrules as pr
from quoridor.mcts import BatchedMCTS
from quoridor.net import NetEvaluator, load_checkpoint

MOVE_NAMES = ["N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW"]
NAME_TO_MOVE = {n: i for i, n in enumerate(MOVE_NAMES)}


def describe(action: int) -> str:
    if action >= fr.MOVE_BASE:
        return MOVE_NAMES[action - fr.MOVE_BASE]
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, fr.W)
    return f"{'H' if orient == 0 else 'V'} {wr} {wc}"


def parse(text: str) -> int | None:
    """Parse user input into an action index, or None if unrecognised."""
    parts = text.strip().upper().split()
    if not parts:
        return None
    if parts[0] in NAME_TO_MOVE and len(parts) == 1:
        return fr.MOVE_BASE + NAME_TO_MOVE[parts[0]]
    if parts[0] in ("H", "V") and len(parts) == 3:
        try:
            wr, wc = int(parts[1]), int(parts[2])
        except ValueError:
            return None
        if not (0 <= wr < fr.W and 0 <= wc < fr.W):
            return None
        return (0 if parts[0] == "H" else 1) * fr.NUM_WALL_SLOTS + wr * fr.W + wc
    return None


def to_py(st: np.ndarray) -> pr.State:
    s = pr.State()
    s.walls_h = [int(x) for x in st[fr.WH_OFF:fr.WH_OFF + 64]]
    s.walls_v = [int(x) for x in st[fr.WV_OFF:fr.WV_OFF + 64]]
    s.pawns = [int(st[fr.IDX_P0]), int(st[fr.IDX_P1])]
    s.walls_left = [int(st[fr.IDX_WL0]), int(st[fr.IDX_WL1])]
    s.turn = int(st[fr.IDX_TURN])
    return s


class Engine:
    def __init__(self, checkpoint: str, sims: int, device: str, c_puct: float, fpu: float):
        net, self.meta = load_checkpoint(checkpoint, device)
        self.mcts = BatchedMCTS(
            NetEvaluator(net, device=device), n_games=1,
            max_nodes=sims + 64, c_puct=c_puct, fpu_reduction=fpu,
        )
        self.sims = sims

    def analyse(self, st: np.ndarray, top: int = 6):
        visits = self.mcts.search(st.reshape(1, -1).copy(), self.sims)[0]
        value = float(self.mcts.root_value()[0])
        total = visits.sum()
        order = np.argsort(-visits)[:top]
        moves = [
            (int(a), describe(int(a)), float(visits[a]), float(visits[a] / max(total, 1)))
            for a in order
            if visits[a] > 0
        ]
        return value, moves

    def best_move(self, st: np.ndarray) -> int:
        visits = self.mcts.search(st.reshape(1, -1).copy(), self.sims)[0]
        return int(visits.argmax())


def show_analysis(engine: Engine, st: np.ndarray) -> None:
    value, moves = engine.analyse(st)
    side = "A" if int(st[fr.IDX_TURN]) == 0 else "B"
    win_pct = 100 * (value + 1) / 2
    print(f"  eval for {side}: {value:+.3f}  (~{win_pct:.0f}% win)")
    for a, name, n, share in moves:
        print(f"    {name:<8} {n:>6.0f} visits  {100 * share:>5.1f}%")


def cmd_play(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Engine(args.checkpoint, args.sims, device, args.c_puct, args.fpu)
    human = 0 if args.first else 1
    st = fr.initial_state()
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)

    print(__doc__)
    print(f"You are {'A (top, racing to row 8)' if human == 0 else 'B (bottom, racing to row 0)'}")
    print(f"Engine: {args.checkpoint} at {args.sims} sims/move")
    print("Commands: a move, 'hint', 'eval', or 'quit'\n")

    while True:
        w = fr.winner(st)
        if w >= 0:
            print(to_py(st).render())
            print(f"\n{'You win!' if w == human else 'Engine wins.'}")
            return
        print(to_py(st).render())

        if int(st[fr.IDX_TURN]) == human:
            fr.legal_mask(st, mask, scratch)
            while True:
                raw = input("your move> ").strip()
                if raw.lower() in ("quit", "exit"):
                    return
                if raw.lower() == "eval":
                    show_analysis(engine, st)
                    continue
                if raw.lower() == "hint":
                    print(f"  engine suggests: {describe(engine.best_move(st))}")
                    continue
                action = parse(raw)
                if action is None:
                    print("  could not parse; try 'N', 'SS', 'H 3 4', 'hint', 'eval', 'quit'")
                    continue
                if not mask[action]:
                    print("  that move is not legal here")
                    continue
                break
            fr.apply_action(st, action, scratch)
        else:
            action = engine.best_move(st)
            print(f"engine plays: {describe(action)}")
            if args.show_eval:
                show_analysis(engine, st)
            fr.apply_action(st, action, scratch)
        print()


def cmd_analyse(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Engine(args.checkpoint, args.sims, device, args.c_puct, args.fpu)
    st = fr.initial_state()
    scratch = fr.make_scratch()

    if args.moves:
        for tok in args.moves.split(","):
            a = parse(tok)
            if a is None:
                raise SystemExit(f"could not parse move {tok!r}")
            fr.apply_action(st, a, scratch)

    print(to_py(st).render())
    print()
    show_analysis(engine, st)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play or analyse with the Quoridor engine")
    ap.add_argument("checkpoint")
    ap.add_argument("--sims", type=int, default=800, help="search simulations per move")
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--fpu", type=float, default=0.2)
    sub = ap.add_subparsers(dest="mode")

    p = sub.add_parser("play", help="play a game against the engine")
    p.add_argument("--first", action="store_true", help="you move first")
    p.add_argument("--show-eval", action="store_true", help="show engine eval each move")
    p.set_defaults(func=cmd_play)

    a = sub.add_parser("analyse", help="analyse a position")
    a.add_argument("--moves", default="", help="comma-separated moves from the start")
    a.set_defaults(func=cmd_analyse)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        args.func = cmd_play
        args.first = True
        args.show_eval = False
    args.func(args)


if __name__ == "__main__":
    main()
