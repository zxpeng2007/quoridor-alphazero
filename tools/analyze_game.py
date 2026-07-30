"""Analyse a barricade.gg game with the trained engine.

Fetches the game record from the public API, decodes the site's move notation
into engine actions, replays the game, and reports where each side went wrong:
per-move evaluation, blunders, and what the engine would have played instead.

Notation
--------
Pawn moves are the destination square, ``<file><rank>`` with file a-i and rank
1-9. Walls are ``h``/``v`` plus a square, ``h<file><rank>`` / ``v<file><rank>``,
with file a-h and rank 1-8 (the 8x8 slot grid).

The mapping from site coordinates to engine coordinates is *derived*, not
assumed: :func:`decode_game` tries the plausible conventions and keeps the one
under which every move in the game is legal. A wrong convention produces an
illegal move within a few plies, so a full legal replay is strong evidence the
mapping is right.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS
from quoridor.net import NetEvaluator, load_checkpoint

API = "https://api.barricade.gg/games/{}"
FILES = "abcdefghi"

# Candidate conventions: (rank_to_row, wall_row_offset, wall_col_offset).
# rank_to_row "up": row = rank - 1;  "down": row = 9 - rank.
CONVENTIONS = [
    ("up", 0, 0), ("up", -1, 0), ("up", 0, -1), ("up", -1, -1),
    ("down", 0, 0), ("down", -1, 0), ("down", 0, -1), ("down", -1, -1),
]


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def fetch(share_code: str) -> dict:
    """Fetch the game record. The API rejects requests without a browser UA."""
    req = urllib.request.Request(
        API.format(share_code),
        headers={"User-Agent": UA, "Accept": "application/json",
                 "Origin": "https://barricade.gg", "Referer": "https://barricade.gg/"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def parse_token(tok: str) -> tuple[str, int, int]:
    """``('pawn'|'h'|'v', file_index, rank)``.

    Walls are exactly three characters (``h``/``v`` + file + rank); pawn moves
    are two (file + rank). Length disambiguates moves like ``h6``, which is a
    pawn stepping to file h, not a malformed wall.
    """
    tok = tok.strip()
    if len(tok) == 3 and tok[0] in "hv":
        return tok[0], FILES.index(tok[1]), int(tok[2])
    return "pawn", FILES.index(tok[0]), int(tok[1:])


def decode_game(tokens: list[str]) -> tuple[list[int], list[np.ndarray], str]:
    """Replay under each candidate convention; return the one that is fully legal."""
    best: tuple[int, list, list, str] | None = None
    for direction, dwr, dwc in CONVENTIONS:
        st = fr.initial_state()
        scratch = fr.make_scratch()
        mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
        dests = np.zeros(12, dtype=np.int32)
        actions: list[int] = []
        states: list[np.ndarray] = [st.copy()]
        ok = True
        for tok in tokens:
            kind, f, rank = parse_token(tok)
            row = rank - 1 if direction == "up" else 9 - rank
            fr.legal_mask_dests(st, mask, scratch, dests)
            action = -1
            if kind == "pawn":
                target = row * 9 + f
                for i in range(12):
                    if mask[fr.MOVE_BASE + i] and int(dests[i]) == target:
                        action = fr.MOVE_BASE + i
                        break
            else:
                wr = (rank - 1 if direction == "up" else 8 - rank) + dwr
                wc = f + dwc
                if 0 <= wr < 8 and 0 <= wc < 8:
                    base = 0 if kind == "h" else fr.NUM_WALL_SLOTS
                    cand = base + wr * 8 + wc
                    if mask[cand]:
                        action = cand
            if action < 0:
                ok = False
                break
            fr.apply_action(st, action, scratch)
            actions.append(action)
            states.append(st.copy())
        label = f"{direction}/wall{dwr:+d}{dwc:+d}"
        if ok:
            return actions, states, label
        if best is None or len(actions) > best[0]:
            best = (len(actions), actions, states, label)
    raise SystemExit(
        f"could not decode the game under any convention; best was "
        f"{best[3]} reaching ply {best[0]}/{len(tokens)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game", help="share code, e.g. b8ewfg")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--sims", type=int, default=8192)
    ap.add_argument("--leaf", type=int, default=32,
                    help="leaves per network call (1 = exact sequential search)")
    ap.add_argument("--blunder", type=float, default=0.35,
                    help="eval drop that counts as a mistake")
    args = ap.parse_args()

    rec = fetch(args.game)
    tokens = rec["historyCsv"].split(",")
    p1, p2 = rec["player1Username"], rec["player2Username"]
    print(f"{p1} ({rec['p1Rating']:.0f}) vs {p2} ({rec['p2Rating']:.0f})")
    print(f"  {'ranked' if rec.get('isRanked') else 'casual'}, "
          f"{rec['timeInitial']}+{rec['timeIncrement']}, "
          f"{len(tokens)} moves, winner: "
          f"{p1 if rec['winner'] == '1' else p2} by {rec['finishedReason']}")
    print(f"  barricades left: {p1} {rec['p1RemainingBarricades']}, "
          f"{p2} {rec['p2RemainingBarricades']}")

    actions, states, label = decode_game(tokens)
    print(f"  decoded cleanly under convention: {label}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()
    net, meta = load_checkpoint(args.checkpoint, device)
    shapes = (1,) if args.leaf <= 1 else (1, args.leaf)
    ev = NetEvaluator(net, device=device, graph_batches=shapes)
    mcts = BatchedMCTS(ev, n_games=1, max_nodes=args.sims + 128,
                       leaf_batch=args.leaf)
    print(f"engine: {meta.get('checkpoint', args.checkpoint)} "
          f"(gen {meta.get('iteration', '?')}) at {args.sims} sims/move")

    # Site player 1 moves first, which is engine player 0.
    names = {0: p1, 1: p2}
    scratch = fr.make_scratch()
    rows = []
    prev_eval_for: dict[int, float] = {}

    for ply, action in enumerate(actions):
        st = states[ply]
        mover = int(st[fr.IDX_TURN])
        visits = mcts.search(st.reshape(1, -1).copy(), args.sims)[0]
        value = float(mcts.root_value()[0])  # mover's perspective
        best_action = int(visits.argmax())
        played_share = float(visits[action] / max(visits.sum(), 1))

        after = states[ply + 1]
        d_self = fr.shortest_path_len(after, mover, scratch)
        d_opp = fr.shortest_path_len(after, 1 - mover, scratch)

        rows.append({
            "ply": ply, "mover": mover, "token": tokens[ply],
            "eval": value, "best": best_action, "played": action,
            "share": played_share, "d_self": d_self, "d_opp": d_opp,
        })
        prev_eval_for[mover] = value

    # Second pass: a mistake is a drop in *that player's* eval between their
    # own consecutive turns.
    print(f"\n{'ply':>4} {'who':<11} {'move':<6} {'eval':>7} {'drop':>7} "
          f"{'played%':>8}  {'dist':>7}  note")
    print("-" * 78)
    last_for: dict[int, float] = {}
    mistakes = []
    for r in rows:
        mover, val = r["mover"], r["eval"]
        drop = val - last_for.get(mover, val)
        last_for[mover] = val
        note = ""
        if drop <= -args.blunder:
            note = "MISTAKE"
            if r["best"] != r["played"]:
                note += f" (engine: {describe(r['best'])})"
            mistakes.append((r, drop))
        elif r["share"] < 0.02 and r["best"] != r["played"]:
            note = f"engine prefers {describe(r['best'])}"
        print(f"{r['ply'] + 1:>4} {names[mover][:11]:<11} {r['token']:<6} "
              f"{val:>+7.2f} {drop:>+7.2f} {100 * r['share']:>7.1f}%  "
              f"{r['d_self']:>3}v{r['d_opp']:<3}  {note}")

    print(f"\n{len(mistakes)} move(s) lost {args.blunder:+.2f} or more:")
    for r, drop in mistakes:
        print(f"  ply {r['ply'] + 1} ({names[r['mover']]}) {r['token']}: "
              f"{drop:+.2f} -> eval {r['eval']:+.2f}; "
              f"engine wanted {describe(r['best'])}")

    times = rec.get("moveTimes") or []
    if times:
        spent = {0: [], 1: []}
        prev = {0: rec["timeInitial"] * 1000, 1: rec["timeInitial"] * 1000}
        inc = rec["timeIncrement"] * 1000
        for i, remaining in enumerate(times):
            who = i % 2
            spent[who].append(prev[who] + inc - remaining)
            prev[who] = remaining
        for who in (0, 1):
            s = spent[who]
            if s:
                print(f"\n{names[who]} think time: mean {np.mean(s) / 1000:.1f}s, "
                      f"median {np.median(s) / 1000:.1f}s, max {max(s) / 1000:.1f}s")


def describe(action: int) -> str:
    if action >= fr.MOVE_BASE:
        names = ["N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW"]
        return f"pawn {names[action - fr.MOVE_BASE]}"
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, 8)
    return f"wall {'h' if orient == 0 else 'v'}{FILES[wc]}{wr + 1}"


if __name__ == "__main__":
    main()
