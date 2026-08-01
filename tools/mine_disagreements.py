"""Mine positions where the value head is confidently wrong, and label them.

Why this exists
---------------
Game qj4ec3 was lost to a wall move (``he3``) that the engine ranked ~24th. The
cause was not search depth -- at 600k simulations that move's share of the search
*fell* to 0.01% -- but the value head: on the position it creates, the raw network
says +0.354 for us while a deep search says -0.408. PUCT then correctly prunes a
branch whose value estimate is simply wrong, and more simulations only reinforce
the error.

Self-play cannot repair this on its own. Every label in that loop is produced by
the same network that holds the misconception: games run at 256 simulations, and
when such a move *is* explored the continuation is played by two copies of the
blind network, so even the game result tends to confirm the error. Six
generations (gen-010..gen-016) all rank ``he3`` between 14th and 36th.

So this tool breaks the loop with the one signal that is demonstrably stronger
than the raw value head: deep search. On the position above, search converges to
the truth -- the sign flips at ~2k simulations and reaches -0.408 by 200k -- and
the game result agrees with it.

What it does
------------
1. Collects positions from real ranked games (human opponents play the structures
   self-play under-generates).
2. Expands each position by legal *replies*, weighted towards low-prior moves,
   because that is precisely where the blindness lives.
3. Screens cheaply: raw network value vs a short search.
4. Labels survivors with a deep search: value from the root, policy from the
   visit distribution.
5. Keeps only genuine disagreements, and holds a slice out so the repair can be
   measured (the promotion gate runs at 200 simulations and cannot see it).

Targets follow the replay-buffer conventions: value is from the perspective of
the player to move, policy is in *real* action space (``ReplayBuffer.sample``
applies mirroring and canonicalisation itself).

    python tools/mine_disagreements.py --user donked --out data/mined.npz
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS
from quoridor.net import NetEvaluator, load_checkpoint

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "Gecko/20100101 Firefox/128.0"),
    "Accept": "application/json",
    "Origin": "https://barricade.gg",
    "Referer": "https://barricade.gg/",
}
USER_GAMES = "https://api.barricade.gg/api/users/{}/games?page={}&limit=50"
GAME = "https://api.barricade.gg/games/{}"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def game_codes(user: str, max_games: int) -> list[str]:
    """Share codes for a user's games, newest first."""
    codes, page = [], 1
    while len(codes) < max_games:
        try:
            blob = _get(USER_GAMES.format(user, page))
        except urllib.error.HTTPError as e:
            print(f"  history page {page}: HTTP {e.code}, stopping")
            break
        rows = blob.get("games") or blob.get("data") or blob.get("items") or []
        if isinstance(blob, list):
            rows = blob
        if not rows:
            break
        for row in rows:
            code = row.get("shareCode") or row.get("share_code") or row.get("code")
            if code:
                codes.append(code)
        page += 1
    return codes[:max_games]


def positions_from_game(code: str) -> list[np.ndarray]:
    """Every position of a game, replayed through the rules engine."""
    from analyze_game import decode_game  # local import: tools/ on sys.path

    rec = _get(GAME.format(code))
    csv = rec.get("historyCsv")
    if not csv:
        return []
    _, states, _ = decode_game(csv.split(","))
    return [s.copy() for s in states]


class Miner:
    def __init__(self, checkpoint: str, device: str, screen_sims: int,
                 label_sims: int, batch: int, label_batch: int):
        torch.backends.cudnn.benchmark = False   # reproducible across processes
        fr.warmup()
        self.net, self.meta = load_checkpoint(checkpoint, device)
        self.device = device
        self.batch = batch
        self.label_batch = label_batch
        self.screen_sims = screen_sims
        self.label_sims = label_sims
        self.ev = NetEvaluator(self.net, device=device,
                               graph_batches=(batch, label_batch))
        # Trees are preallocated, so the deep pass needs a smaller game count:
        # nodes x games x ~2.5 KB, and the deep pass has ~20x the nodes.
        self.screen = BatchedMCTS(self.ev, n_games=batch,
                                  max_nodes=screen_sims + 256, leaf_batch=1)
        self.label = BatchedMCTS(self.ev, n_games=label_batch,
                                 max_nodes=label_sims + 256, leaf_batch=1)
        self.scratch = fr.make_scratch()

    def raw_and_prior(self, states: np.ndarray):
        """Network value and prior for states, no search at all.

        Chunked to ``self.batch``: the search object's encode buffers are sized
        to its game count, so handing it a larger array would run off the end.
        """
        n = len(states)
        legal = np.zeros((n, fr.NUM_ACTIONS), dtype=np.uint8)
        for i in range(n):
            fr.legal_mask(states[i], legal[i], self.scratch)
        vals = np.zeros(n, dtype=np.float32)
        pols = np.zeros((n, fr.NUM_ACTIONS), dtype=np.float32)
        for s in range(0, n, self.batch):
            e = min(s + self.batch, n)
            # Reuse the search's own encoder so the planes match exactly.
            p, v = self.screen._evaluate(
                np.ascontiguousarray(states[s:e]), legal[s:e])
            pols[s:e], vals[s:e] = p, v
        return vals, pols, legal

    def _run(self, mcts: BatchedMCTS, states: np.ndarray, sims: int, batch: int):
        """Search a list of states in chunks, returning (values, visit counts)."""
        n = len(states)
        vals = np.zeros(n, dtype=np.float32)
        visits = np.zeros((n, fr.NUM_ACTIONS), dtype=np.float32)
        for s in range(0, n, batch):
            chunk = states[s:s + batch]
            v = mcts.search(np.ascontiguousarray(chunk), sims)
            vals[s:s + len(chunk)] = mcts.root_value()[:len(chunk)]
            visits[s:s + len(chunk)] = v[:len(chunk)]
        return vals, visits

    def children_of(self, state: np.ndarray, prior: np.ndarray,
                    legal: np.ndarray, keep: int, rng) -> list[np.ndarray]:
        """Replies to a position, biased towards the ones search ignores.

        The top few by prior are the lines the engine already understands; the
        blind spot is in the tail, so most of the sample is drawn from there.
        """
        idx = np.nonzero(legal)[0]
        if len(idx) == 0:
            return []
        order = idx[np.argsort(-prior[idx])]
        head, tail = order[:3], order[3:]
        n_tail = max(0, keep - len(head))
        if len(tail) > n_tail:
            tail = rng.choice(tail, size=n_tail, replace=False)
        out = []
        for a in np.concatenate([head, tail]).astype(int):
            child = state.copy()
            fr.apply_action(child, int(a), self.scratch)
            if fr.winner(child) < 0:          # finished positions teach nothing
                out.append(child)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--user", default="donked")
    ap.add_argument("--games", default="", help="explicit share codes, comma separated")
    ap.add_argument("--max-games", type=int, default=80)
    ap.add_argument("--children", type=int, default=16,
                    help="replies sampled per game position")
    ap.add_argument("--screen-sims", type=int, default=1500)
    ap.add_argument("--label-sims", type=int, default=50_000)
    ap.add_argument("--screen-gap", type=float, default=0.25,
                    help="|raw - screen| needed to spend a deep search")
    ap.add_argument("--keep-gap", type=float, default=0.40,
                    help="|raw - deep| needed to keep the position")
    ap.add_argument("--batch", type=int, default=256,
                    help="positions searched at once during screening")
    ap.add_argument("--label-batch", type=int, default=64,
                    help="positions searched at once during deep labelling; the "
                         "deep trees are ~20x larger, so this must be smaller")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--out", default="data/mined.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()

    codes = ([c.strip() for c in args.games.split(",") if c.strip()]
             or game_codes(args.user, args.max_games))
    print(f"{len(codes)} games")

    m = Miner(args.checkpoint, args.device, args.screen_sims,
              args.label_sims, args.batch, args.label_batch)
    print(f"engine: gen {m.meta.get('iteration', '?')} "
          f"({m.meta.get('elo', 0):+.0f} Elo)\n")

    # 1. collect candidate children from every game position
    cands: list[np.ndarray] = []
    for i, code in enumerate(codes, 1):
        try:
            states = positions_from_game(code)
        except Exception as e:
            print(f"  {code}: {type(e).__name__}, skipped")
            continue
        for st in states:
            if fr.winner(st) >= 0:
                continue
            val, pol, legal = m.raw_and_prior(st.reshape(1, -1))
            cands.extend(m.children_of(st, pol[0], legal[0], args.children, rng))
        print(f"  [{i}/{len(codes)}] {code}: {len(states)} plies, "
              f"{len(cands):,} candidates so far", flush=True)
    if not cands:
        print("no candidates; nothing to do")
        return
    cands = np.stack(cands)
    print(f"\n{len(cands):,} candidate positions")

    # 2. cheap screen: raw value vs a short search
    raw, _, _ = m.raw_and_prior(cands)
    t0 = time.perf_counter()
    screen, _ = m._run(m.screen, cands, args.screen_sims, args.batch)
    gap = np.abs(raw - screen)
    sel = np.nonzero(gap >= args.screen_gap)[0]
    print(f"screen ({args.screen_sims:,} sims, {time.perf_counter()-t0:.0f}s): "
          f"{len(sel):,} of {len(cands):,} exceed {args.screen_gap} "
          f"(median gap {np.median(gap):.3f})")
    if len(sel) == 0:
        print("nothing survived the screen")
        return

    # 3. deep labels for the survivors
    t0 = time.perf_counter()
    deep_v, deep_visits = m._run(m.label, cands[sel], args.label_sims,
                                args.label_batch)
    print(f"label ({args.label_sims:,} sims, {time.perf_counter()-t0:.0f}s): done")

    final_gap = np.abs(raw[sel] - deep_v)
    keep = np.nonzero(final_gap >= args.keep_gap)[0]
    print(f"kept {len(keep):,} with |raw - deep| >= {args.keep_gap} "
          f"(max {final_gap.max():.3f}, median {np.median(final_gap):.3f})")
    if len(keep) == 0:
        print("no genuine disagreements found")
        return

    states_out = cands[sel][keep]
    values_out = deep_v[keep].astype(np.float32)
    visits = deep_visits[keep]
    policies_out = (visits / np.maximum(visits.sum(axis=1, keepdims=True), 1e-9)
                    ).astype(np.float16)

    # 4. split: the holdout is what makes the repair measurable
    perm = rng.permutation(len(states_out))
    n_hold = int(len(perm) * args.holdout)
    hold, train = perm[:n_hold], perm[n_hold:]
    out = args.out
    hold_path = out.replace(".npz", "_holdout.npz")
    np.savez_compressed(out, states=states_out[train], policies=policies_out[train],
                        values=values_out[train], raw=raw[sel][keep][train])
    np.savez_compressed(hold_path, states=states_out[hold],
                        policies=policies_out[hold], values=values_out[hold],
                        raw=raw[sel][keep][hold])
    print(f"\nwrote {len(train):,} -> {out}")
    print(f"wrote {len(hold):,} -> {hold_path}")
    print(f"raw-vs-deep MAE on holdout at mining time: "
          f"{np.abs(raw[sel][keep][hold] - values_out[hold]).mean():.3f}")
    print(f"total {(time.perf_counter()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
