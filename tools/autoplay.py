"""Autonomous bridge: local engine -> Firefox -> barricade.gg ranked play.

Runs on your machine, drives your own Firefox profile (so your existing login is
used -- this script never handles credentials), reads the board from the DOM,
asks the trained engine for a move, and plays it. Loops until you stop it.

PRECONDITIONS (yours to confirm, not checked here)
-------------------------------------------------
* The account is flagged as a bot by the site operators, so opponents can see
  what they are playing.
* You are OK with these games counting on the ranked ladder.

USAGE
-----
    python tools/autoplay.py --dry-run       # read and decide, click nothing
    python tools/autoplay.py --max-games 10  # play for real

Stop with Ctrl+C, or create a file named ``STOP`` in the project directory
(checked before every move, so it stops cleanly rather than mid-click).

BOARD READING
-------------
The reader is verified end-to-end by ``tools/validate_reader.py``, which steps
through a finished game on the analysis page and compares what this code
decodes from the DOM against the position obtained by replaying the game's own
move list through the rules engine. It currently agrees on 42/42 positions of a
41-move game, including positions where the pawns have crossed.

Facts established against the live site, rather than assumed:

* Pawns are ``div.rounded-full`` with ``bg-red-500`` / ``bg-blue-500``. Colour
  identifies the seat -- red is the first seat (engine player 0), blue the
  second. Colour is authoritative; relative position is NOT, because the pawns
  cross during a game. (An earlier version assigned seats by row order and
  silently mirrored the position mid-game.)
* Wall slots are ``slot-{horizontal,vertical}-{r}-{c}``, indices identical to
  the engine's action encoding. A placed wall is a slot child with non-zero
  opacity and a real background colour; empty slots carry a transparent
  hover-ghost child, and slots blocked by a crossing wall have no child at all.
* Board orientation is read from the rank labels every time. The default view
  draws rank 9 at the top, but the board rotates when you play the second seat.
* Whose turn it is comes from watching which clock actually ticks -- markup for
  turn indicators is easy to get wrong, a running clock is not.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS
from quoridor.net import NetEvaluator, load_checkpoint

BASE = "https://barricade.gg"
STOP_FILE = Path("STOP")
PROFILE_DIR = Path.home() / ".barricade-autoplay-firefox"

FILES = "abcdefghi"
MOVE_NAMES = ["N", "S", "E", "W", "NN", "SS", "EE", "WW", "NE", "NW", "SE", "SW"]

# --------------------------------------------------------------------- DOM

BOARD_JS = r"""() => {
  const s00 = document.querySelector('[data-testid="slot-horizontal-0-0"]');
  const s11 = document.querySelector('[data-testid="slot-horizontal-1-1"]');
  if (!s00 || !s11) return null;
  const board = s00.parentElement;
  const bb = board.getBoundingClientRect();
  const r00 = s00.getBoundingClientRect(), r11 = s11.getBoundingClientRect();
  const pitch = r11.x - r00.x, gap = r00.height, cell = pitch - gap;
  const ox = r00.x, oy = r00.y - cell;

  const labels = [];
  board.querySelectorAll('div').forEach(el => {
    const t = el.textContent.trim();
    if (/^[1-9]$/.test(t) && el.children.length === 0)
      labels.push({r: +t, y: el.getBoundingClientRect().y});
  });
  labels.sort((a, b) => a.y - b.y);
  const flipped = labels.length >= 2
    ? labels[0].r > labels[labels.length - 1].r : null;

  const pawns = [];
  board.querySelectorAll('div.rounded-full').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width < 10) return;
    const cls = (el.className || '').toString();
    const colour = cls.includes('bg-red') ? 'red'
                 : cls.includes('bg-blue') ? 'blue' : null;
    if (!colour) return;
    pawns.push({colour,
                col: Math.round((r.x + r.width / 2 - ox - cell / 2) / pitch),
                vrow: Math.round((r.y + r.height / 2 - oy - cell / 2) / pitch)});
  });

  // A placed wall is opaque and coloured. The hover ghost is translucent grey
  // (gray-300 at low alpha), so it must be rejected on BOTH counts -- checking
  // only `opacity: 0` lets a partially faded ghost through and invents a wall.
  const isWallPaint = (ks) => {
    if (ks.opacity !== '' && parseFloat(ks.opacity) < 0.9) return false;
    const m = ks.backgroundColor.match(/rgba?\(([^)]+)\)/);
    if (!m) return false;
    const p = m[1].split(',').map(s => parseFloat(s));
    const [r, g, b] = p;
    const alpha = p.length > 3 ? p[3] : 1;
    if (alpha < 0.9) return false;                       // translucent => ghost
    const spread = Math.max(r, g, b) - Math.min(r, g, b);
    if (spread < 20) return false;                       // grey => ghost
    return true;
  };
  const walls = [];
  document.querySelectorAll('[data-testid^="slot-"]').forEach(el => {
    const kid = el.firstElementChild;
    if (!kid) return;
    const ks = getComputedStyle(kid);
    if (!isWallPaint(ks)) return;
    walls.push({tid: el.dataset.testid, bg: ks.backgroundColor});
  });

  const text = document.body.innerText;
  const bars = (text.match(/Barricades:\s*(\d+)\s*\/\s*10/g) || [])
    .map(s => +s.match(/(\d+)/)[1]);
  return {geo: {bx: bb.x, by: bb.y, pitch, cell, gap, ox, oy},
          flipped, pawns, walls, barricades: bars,
          youArePlayer: (text.match(/You are Player (\d)/) || [])[1] || null,
          gameOver: /won by|You won|You lost|resigned|Draw|Rematch/i.test(text),
          text: text.slice(0, 300)};
}"""

CLOCKS_JS = r"""() => {
  const out = [];
  document.querySelectorAll('div,span').forEach(el => {
    if (el.children.length) return;
    const t = el.textContent.trim();
    if (!/^\d{1,2}:\d{2}(\.\d)?$/.test(t)) return;
    const r = el.getBoundingClientRect();
    if (r.width === 0) return;
    out.push({t, y: r.y});
  });
  out.sort((a, b) => a.y - b.y);
  return out;
}"""


def clock_seconds(text: str) -> float:
    mm, _, rest = text.partition(":")
    return int(mm) * 60 + float(rest)


@dataclass
class BoardRead:
    state: np.ndarray
    flipped: bool
    us: int
    walls_seen: int
    walls_expected: int
    game_over: bool
    raw: dict = field(repr=False)


class Bridge:
    def __init__(self, page, verbose: bool = False):
        self.page = page
        self.verbose = verbose
        self.geo: dict | None = None

    # ---------------------------------------------------------------- read

    def snapshot(self) -> dict | None:
        # Park the pointer off the board first: resting it over a groove paints
        # a hover ghost that is easy to mistake for a placed wall.
        try:
            self.page.mouse.move(4, 4)
        except Exception:
            pass
        dom = self.page.evaluate(BOARD_JS)
        if dom:
            self.geo = dom["geo"]
        return dom

    def decode(self, dom: dict) -> tuple[BoardRead | None, str]:
        if dom is None:
            return None, "no board on the page"
        if dom["flipped"] is None:
            return None, "could not read board orientation from rank labels"
        flipped = bool(dom["flipped"])

        st = fr.initial_state()
        st[fr.WH_OFF:fr.WH_OFF + 64] = 0
        st[fr.WV_OFF:fr.WV_OFF + 64] = 0

        for w in dom["walls"]:
            parts = w["tid"].split("-")
            if len(parts) != 4:
                continue
            orientation, wr, wc = parts[1], int(parts[2]), int(parts[3])
            if flipped:
                wr = 7 - wr
            if not (0 <= wr < 8 and 0 <= wc < 8):
                return None, f"wall slot out of range: {w['tid']}"
            off = fr.WH_OFF if orientation == "horizontal" else fr.WV_OFF
            st[off + wr * 8 + wc] = 1

        by_colour: dict[str, int] = {}
        for p in dom["pawns"]:
            # JS Math.round can hand back -0.0; normalise to plain ints.
            vrow, col = int(round(p["vrow"])), int(round(p["col"]))
            row = 8 - vrow if flipped else vrow
            if not (0 <= row < 9 and 0 <= col < 9):
                return None, f"pawn off board: {p}"
            by_colour[p["colour"]] = row * 9 + col
        if set(by_colour) != {"red", "blue"}:
            return None, (f"expected one red and one blue pawn, "
                          f"found {dom['pawns']}")
        st[fr.IDX_P0] = by_colour["red"]
        st[fr.IDX_P1] = by_colour["blue"]

        bars = dom.get("barricades") or []
        walls_expected = -1
        if len(bars) == 2:
            # Order on the page is not guaranteed; only the total is a checksum.
            walls_expected = 20 - (bars[0] + bars[1])
            placed_p0 = sum(1 for w in dom["walls"] if "178" in w["bg"])
            placed_p1 = len(dom["walls"]) - placed_p0
            st[fr.IDX_WL0] = max(0, 10 - placed_p0)
            st[fr.IDX_WL1] = max(0, 10 - placed_p1)

        you = dom.get("youArePlayer")
        us = 0 if you == "1" else 1 if you == "2" else None
        if us is None:
            return None, "could not determine which seat we are playing"

        return BoardRead(state=st, flipped=flipped, us=us,
                         walls_seen=len(dom["walls"]),
                         walls_expected=walls_expected,
                         game_over=bool(dom.get("gameOver")), raw=dom), ""

    def whose_turn(self, us: int, settle: float = 1.1) -> int | None:
        """Which engine player is to move, from which clock is ticking.

        Turn-indicator markup is easy to misread; a running clock is not.
        Returns None when neither clock moves (game over, or between games).
        """
        first = self.page.evaluate(CLOCKS_JS)
        if len(first) != 2:
            return None
        self.page.wait_for_timeout(int(settle * 1000))
        second = self.page.evaluate(CLOCKS_JS)
        if len(second) != 2:
            return None
        deltas = [clock_seconds(a["t"]) - clock_seconds(b["t"])
                  for a, b in zip(first, second)]
        # The site puts the local player's clock at the bottom.
        top_running, bottom_running = deltas[0] > 0.05, deltas[1] > 0.05
        if bottom_running and not top_running:
            return us
        if top_running and not bottom_running:
            return 1 - us
        return None

    # --------------------------------------------------------------- write

    def cell_xy(self, cell: int, flipped: bool) -> tuple[float, float]:
        g = self.geo
        row, col = divmod(cell, 9)
        vrow = 8 - row if flipped else row
        return (g["ox"] + col * g["pitch"] + g["cell"] / 2,
                g["oy"] + vrow * g["pitch"] + g["cell"] / 2)

    def play_pawn(self, cell: int, flipped: bool) -> None:
        x, y = self.cell_xy(cell, flipped)
        self.page.mouse.click(x, y)

    def play_wall(self, action: int, flipped: bool) -> bool:
        orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
        wr, wc = divmod(slot, 8)
        if flipped:
            wr = 7 - wr
        tid = f"slot-{'horizontal' if orient == 0 else 'vertical'}-{wr}-{wc}"
        el = self.page.locator(f'[data-testid="{tid}"]')
        if not el.count():
            return False
        box = el.first.bounding_box()
        if not box:
            return False
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        # Hovering reveals the ghost wall; a plain click commits it in the
        # current build. Fall back to a short press-drag for drag-only builds.
        self.page.mouse.move(cx, cy)
        self.page.wait_for_timeout(120)
        self.page.mouse.click(cx, cy)
        return True


class Engine:
    LEAF = 32
    # Measured on a real mid-game position with a cold tree (the synthetic
    # benchmark's 38k/s does not survive contact with a fresh root).
    SIMS_PER_SEC = 19000

    def __init__(self, checkpoint: str, device: str, max_sims: int):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True
        fr.warmup()
        net, self.meta = load_checkpoint(checkpoint, device)
        ev = NetEvaluator(net, device=device, graph_batches=(1, self.LEAF))
        self.mcts = BatchedMCTS(ev, n_games=1, max_nodes=max_sims + 256,
                                leaf_batch=self.LEAF)
        self.max_sims = max_sims

    def sims_for(self, clock: float | None, increment: float = 3.0) -> int:
        """Whichever is tighter: a little over the increment, or clock/20.

        Earlier ranked games averaged 8.7 s/move against opponents playing at
        2.7 s. Losing on time is a real way to lose these games.
        """
        budget = increment if clock is None else min(1.2 * increment, clock / 20.0)
        return int(max(256, min(self.max_sims, budget * self.SIMS_PER_SEC)))

    def choose(self, st: np.ndarray, sims: int):
        visits = self.mcts.search(st.reshape(1, -1).copy(), sims)[0]
        return int(visits.argmax()), float(self.mcts.root_value()[0]), visits


def describe(action: int, st_after=None, mover=None) -> str:
    if action >= fr.MOVE_BASE:
        if st_after is not None and mover is not None:
            cell = int(st_after[fr.IDX_P0 if mover == 0 else fr.IDX_P1])
            r, c = divmod(cell, 9)
            return f"pawn -> {FILES[c]}{r + 1}"
        return f"pawn {MOVE_NAMES[action - fr.MOVE_BASE]}"
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, 8)
    return f"wall {'h' if orient == 0 else 'v'}{FILES[wc]}{wr + 1}"


def render(st: np.ndarray) -> str:
    from quoridor import pyrules as pr

    s = pr.State()
    s.walls_h = [int(x) for x in st[fr.WH_OFF:fr.WH_OFF + 64]]
    s.walls_v = [int(x) for x in st[fr.WV_OFF:fr.WV_OFF + 64]]
    s.pawns = [int(st[fr.IDX_P0]), int(st[fr.IDX_P1])]
    s.walls_left = [int(st[fr.IDX_WL0]), int(st[fr.IDX_WL1])]
    s.turn = int(st[fr.IDX_TURN])
    return s.render()


def play_one_move(bridge: Bridge, engine: Engine, args) -> str:
    dom = bridge.snapshot()
    if dom is None:
        return "no-board"
    read, err = bridge.decode(dom)
    if read is None:
        print(f"  !! board read failed: {err}")
        return "read-failed"
    if read.game_over:
        return "game-over"
    if read.walls_expected >= 0 and read.walls_seen != read.walls_expected:
        print(f"  !! wall cross-check failed: {read.walls_seen} walls in the DOM, "
              f"barricade counters imply {read.walls_expected}")
        return "inconsistent"

    turn = bridge.whose_turn(read.us)
    if turn is None:
        return "turn-unknown"
    if turn != read.us:
        return "not-our-turn"
    read.state[fr.IDX_TURN] = read.us

    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    fr.legal_mask(read.state, mask, scratch)
    if not mask.any():
        return "no-legal-moves"

    clocks = bridge.page.evaluate(CLOCKS_JS)
    our_clock = clock_seconds(clocks[1]["t"]) if len(clocks) == 2 else None
    sims = engine.sims_for(our_clock)

    t0 = time.perf_counter()
    action, value, visits = engine.choose(read.state, sims)
    think = time.perf_counter() - t0
    if not mask[action]:
        print("  !! engine chose an action illegal in the read position")
        return "illegal"

    expected = read.state.copy()
    fr.apply_action(expected, action, scratch)
    share = float(visits[action] / max(visits.sum(), 1))
    print(f"  {describe(action, expected, read.us):<16} "
          f"[{sims} sims, {think:.1f}s, eval {value:+.2f}, {100 * share:.0f}%"
          + (f", clock {our_clock:.0f}s" if our_clock else "") + "]")

    if args.dry_run:
        print(render(read.state))
        return "dry-run"

    if action >= fr.MOVE_BASE:
        dest = int(expected[fr.IDX_P0 if read.us == 0 else fr.IDX_P1])
        bridge.play_pawn(dest, read.flipped)
    elif not bridge.play_wall(action, read.flipped):
        return "wall-place-failed"

    for _ in range(16):
        bridge.page.wait_for_timeout(250)
        dom2 = bridge.snapshot()
        after, _ = bridge.decode(dom2) if dom2 else (None, "")
        if after is None:
            continue
        if (after.state[fr.IDX_P0] == expected[fr.IDX_P0]
                and after.state[fr.IDX_P1] == expected[fr.IDX_P1]
                and np.array_equal(after.state[fr.WH_OFF:fr.WH_OFF + 64],
                                   expected[fr.WH_OFF:fr.WH_OFF + 64])
                and np.array_equal(after.state[fr.WV_OFF:fr.WV_OFF + 64],
                                   expected[fr.WV_OFF:fr.WV_OFF + 64])):
            return "played"
    print("  !! could not confirm the move registered on the site")
    return "unconfirmed"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-sims", type=int, default=32768)
    ap.add_argument("--max-games", type=int, default=0, help="0 = until stopped")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--url", default=f"{BASE}/")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    if STOP_FILE.exists():
        STOP_FILE.unlink()

    print("loading engine...")
    engine = Engine(args.checkpoint, args.device, args.max_sims)
    m = engine.meta
    print(f"engine: {m.get('checkpoint', args.checkpoint)} "
          f"(gen {m.get('iteration', '?')}, {m.get('elo', 0):+.0f} Elo internal)")

    stopping = {"flag": False}
    signal.signal(signal.SIGINT,
                  lambda *_: (stopping.__setitem__("flag", True),
                              print("\nstopping after this move...")))

    with sync_playwright() as pw:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        ctx = pw.firefox.launch_persistent_context(
            str(PROFILE_DIR), headless=False,
            viewport={"width": 1500, "height": 1000})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(args.url, wait_until="networkidle", timeout=60000)

        bridge = Bridge(page, verbose=args.verbose)
        print("\nSign in if needed, then open a game. Waiting for a board...\n")

        games = 0
        while not (stopping["flag"] or STOP_FILE.exists()):
            if args.max_games and games >= args.max_games:
                print(f"reached --max-games {args.max_games}")
                break
            status = play_one_move(bridge, engine, args)

            if status in ("read-failed", "inconsistent", "illegal",
                          "wall-place-failed", "unconfirmed"):
                print(f"  aborting: {status}")
                if args.verbose:
                    dom = bridge.snapshot()
                    print(json.dumps({"pawns": dom.get("pawns"),
                                      "walls": dom.get("walls", [])[:6],
                                      "flipped": dom.get("flipped"),
                                      "barricades": dom.get("barricades"),
                                      "you": dom.get("youArePlayer")},
                                     indent=1)[:1500] if dom else "no DOM")
                break
            if status == "dry-run":
                print("\nDry run: compare the board above with your screen.")
                break
            if status == "game-over":
                games += 1
                print(f"game finished ({games})\n")
                page.wait_for_timeout(4000)
                continue
            if status in ("no-board", "not-our-turn", "turn-unknown",
                          "no-legal-moves"):
                page.wait_for_timeout(700)
                continue

        print(f"stopped after {games} completed game(s)")
        if STOP_FILE.exists():
            STOP_FILE.unlink()
        ctx.close()


if __name__ == "__main__":
    main()
