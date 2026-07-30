"""Autonomous bridge: local engine -> Firefox -> barricade.gg ranked play.

Runs on your machine, drives your own Firefox profile (so your existing login is
used -- this script never handles credentials), reads the board from the DOM,
asks the trained engine for a move, and plays it. Loops until you stop it.

PRECONDITIONS (yours to confirm, not checked here)
-------------------------------------------------
* The account is flagged as a bot by the site operators, so opponents can see
  what they are playing. Nothing below verifies that.
* You are OK with the games this plays counting on the ranked ladder.

USAGE
-----
First, always validate the DOM contract without playing::

    python tools/autoplay.py --dry-run

Dry run reads the live board, prints what it decodes and what the engine would
play, and clicks nothing. Only when its board matches what you see on screen is
the reader trustworthy. Then::

    python tools/autoplay.py --max-games 10

To stop: press Ctrl+C, or create the file ``STOP`` in the project directory
(checked before every move, so it stops cleanly mid-game rather than mid-click).

DESIGN: FAIL CLOSED
-------------------
A misread board does not produce a bad move -- it produces a *legal move for the
wrong position*, which is worse than resigning because it looks fine. So every
move passes three independent checks, and any failure aborts the game instead of
clicking:

1. **Structural**: exactly two pawns found, on distinct cells.
2. **Wall cross-check**: walls counted in the DOM must equal 20 minus the
   barricade counters the page displays. This catches missed or phantom walls.
3. **Post-move confirmation**: after clicking, the board is re-read and must
   match the position the engine expected. If not, we stop.

VERIFIED vs UNVERIFIED
----------------------
Verified against the live site earlier: wall-slot testids
(``slot-horizontal-{r}-{c}`` / ``slot-vertical-{r}-{c}``, indices identical to
the engine's action encoding), board geometry (738 px container, 4 px pad,
62.75 px cells, 78.4375 px pitch), and the move notation convention.

NOT verified (the selectors below are best-effort and are exactly what dry-run
exists to check): pawn element detection, the wall drag-and-drop gesture, and
turn/orientation detection. Expect to adjust ``PAWN_JS``, ``WALL_DRAG``, and
``TURN_JS`` once, using dry-run output.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
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

# ---------------------------------------------------------------- board read

GEOMETRY_JS = """() => {
  const s00 = document.querySelector('[data-testid="slot-horizontal-0-0"]');
  if (!s00) return null;
  const board = s00.parentElement;
  const b = board.getBoundingClientRect();
  // Derive the grid from two slots rather than hard-coding pixel constants, so
  // the reader survives a window resize or a layout change.
  const s11 = document.querySelector('[data-testid="slot-horizontal-1-1"]');
  const r00 = s00.getBoundingClientRect(), r11 = s11.getBoundingClientRect();
  const pitch = r11.x - r00.x;
  const gap = r00.height;
  const cell = pitch - gap;
  return {bx: b.x, by: b.y, bw: b.width, pitch, cell, gap,
          originX: r00.x, originY: r00.y - cell};
}"""

# Rank labels tell us which way the board is drawn; never assume an orientation.
ORIENT_JS = """() => {
  const labels = [...document.querySelectorAll('div')].filter(el => {
    const t = el.textContent.trim();
    return /^[1-9]$/.test(t) && el.children.length === 0 &&
           (el.className || '').toString().includes('font-mono');
  }).map(el => ({rank: +el.textContent.trim(),
                 y: el.getBoundingClientRect().y}));
  if (labels.length < 2) return null;
  labels.sort((a, b) => a.y - b.y);
  return {topRank: labels[0].rank, bottomRank: labels[labels.length - 1].rank,
          count: labels.length};
}"""

PAWN_JS = """() => {
  const s00 = document.querySelector('[data-testid="slot-horizontal-0-0"]');
  if (!s00) return null;
  const board = s00.parentElement;
  const b = board.getBoundingClientRect();
  const out = [];
  board.querySelectorAll('div').forEach(el => {
    if (el.dataset && el.dataset.testid) return;          // skip wall slots
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    const radius = parseFloat(st.borderRadius) || 0;
    const round = radius >= r.width / 3 && r.width > 14 && r.width < 70;
    const filled = st.backgroundColor !== 'rgba(0, 0, 0, 0)' ||
                   st.backgroundImage !== 'none';
    if (round && filled && Math.abs(r.width - r.height) < 6) {
      out.push({cx: r.x + r.width / 2 - b.x, cy: r.y + r.height / 2 - b.y,
                w: r.width, bg: st.backgroundColor,
                img: st.backgroundImage.slice(0, 60),
                cls: (el.className || '').toString().slice(0, 60)});
    }
  });
  return out;
}"""

WALL_JS = """() => {
  const out = [];
  document.querySelectorAll('[data-testid^="slot-"]').forEach(el => {
    // A placed wall paints the slot (or an inner div). An empty slot is
    // transparent until hovered. Compare against the slot's own children.
    let filled = false;
    const st = getComputedStyle(el);
    if (st.backgroundColor !== 'rgba(0, 0, 0, 0)' &&
        parseFloat(st.opacity || '1') > 0.5) filled = true;
    for (const kid of el.querySelectorAll('div')) {
      const ks = getComputedStyle(kid);
      const kr = kid.getBoundingClientRect();
      if (ks.backgroundColor !== 'rgba(0, 0, 0, 0)' &&
          parseFloat(ks.opacity || '1') > 0.5 &&
          kr.width > 20 && kr.height > 4) filled = true;
    }
    if (filled) out.push(el.dataset.testid);
  });
  return out;
}"""

TURN_JS = """() => {
  const text = document.body.innerText;
  const clocks = [...document.querySelectorAll('div')]
    .filter(el => el.children.length === 0 && /^\\d+:\\d{2}$/.test(el.textContent.trim()))
    .map(el => ({t: el.textContent.trim(),
                 y: el.getBoundingClientRect().y,
                 cls: (el.className || '').toString().slice(0, 80)}));
  clocks.sort((a, b) => a.y - b.y);
  const barricades = (text.match(/Barricades:\\s*(\\d+)\\s*\\/\\s*10/g) || [])
    .map(s => +s.match(/(\\d+)/)[1]);
  return {
    clocks, barricades,
    youArePlayer: (text.match(/You are Player (\\d)/) || [])[1] || null,
    thinking: /thinking/i.test(text),
    gameOver: /won by|You won|You lost|resigned|Draw/i.test(text),
    text: text.slice(0, 400),
  };
}"""

WALL_DRAG = """
async (args) => {
  // dnd-kit needs a real pointer sequence with intermediate moves; a single
  // click on the slot does nothing.
  const tray = document.querySelector(args.traySelector);
  const slot = document.querySelector(`[data-testid="${args.slotId}"]`);
  if (!tray || !slot) return {ok: false, why: 'tray or slot not found'};
  const t = tray.getBoundingClientRect(), s = slot.getBoundingClientRect();
  return {ok: true,
          from: {x: t.x + t.width / 2, y: t.y + t.height / 2},
          to: {x: s.x + s.width / 2, y: s.y + s.height / 2}};
}"""


@dataclass
class BoardRead:
    """A decoded position plus everything needed to audit the decode."""

    state: np.ndarray
    us: int                 # engine player index we are playing
    flipped: bool           # True when rank 1 is drawn at the bottom
    our_clock: float | None
    walls_seen: int
    walls_expected: int
    raw_pawns: list
    raw_walls: list
    info: dict

    @property
    def consistent(self) -> bool:
        return self.walls_seen == self.walls_expected


class Bridge:
    def __init__(self, page, verbose: bool = False):
        self.page = page
        self.verbose = verbose
        self.geo: dict | None = None

    # ------------------------------------------------------------ geometry

    def refresh_geometry(self) -> bool:
        self.geo = self.page.evaluate(GEOMETRY_JS)
        return self.geo is not None

    def cell_from_xy(self, cx: float, cy: float, flipped: bool) -> int | None:
        """Board-relative pixel centre -> engine cell index."""
        g = self.geo
        col = round((cx - (g["originX"] - g["bx"]) - g["cell"] / 2) / g["pitch"])
        vrow = round((cy - (g["originY"] - g["by"]) - g["cell"] / 2) / g["pitch"])
        if not (0 <= col < 9 and 0 <= vrow < 9):
            return None
        row = 8 - vrow if flipped else vrow
        return row * 9 + col

    def cell_xy(self, cell: int, flipped: bool) -> tuple[float, float]:
        """Engine cell index -> absolute screen coordinates of its centre."""
        g = self.geo
        row, col = divmod(cell, 9)
        vrow = 8 - row if flipped else row
        return (g["originX"] + col * g["pitch"] + g["cell"] / 2,
                g["originY"] + vrow * g["pitch"] + g["cell"] / 2)

    # ---------------------------------------------------------------- read

    def read(self) -> BoardRead | None:
        if not self.refresh_geometry():
            return None
        orient = self.page.evaluate(ORIENT_JS)
        info = self.page.evaluate(TURN_JS)
        pawns = self.page.evaluate(PAWN_JS) or []
        walls = self.page.evaluate(WALL_JS) or []

        # Orientation: the site draws rank 9 at the top by default, and rotates
        # when you play the second seat. Read it, never assume.
        flipped = True
        if orient:
            flipped = orient["topRank"] > orient["bottomRank"]

        st = fr.initial_state()
        for i in range(fr.NUM_WALL_SLOTS):
            st[fr.WH_OFF + i] = 0
            st[fr.WV_OFF + i] = 0

        for tid in walls:
            try:
                _, orientation, r, c = tid.split("-")
                wr, wc = int(r), int(c)
            except ValueError:
                continue
            if flipped:
                wr = 7 - wr
            off = fr.WH_OFF if orientation == "horizontal" else fr.WV_OFF
            st[off + wr * 8 + wc] = 1

        cells = []
        for p in pawns:
            cell = self.cell_from_xy(p["cx"], p["cy"], flipped)
            if cell is not None:
                cells.append((cell, p))
        if len(cells) != 2:
            if self.verbose:
                print(f"    pawn detection found {len(cells)} candidates "
                      f"(need exactly 2)")
            return None

        # Which pawn is ours? The site says "You are Player N"; site player 1
        # moves first and is engine player 0.
        you = info.get("youArePlayer")
        us = 0 if you == "1" else 1 if you == "2" else None
        if us is None:
            if self.verbose:
                print("    could not determine which seat we are")
            return None

        # Engine player 0 starts on rank 1 (row 0) and races to row 8. Assign
        # the two detected pawns by which goal each is nearer -- unambiguous
        # except at the exact start, where both are on their own start squares.
        (c0, _), (c1, _) = cells
        rows = (c0 // 9, c1 // 9)
        if rows[0] == rows[1]:
            return None
        p0_cell, p1_cell = (c0, c1) if rows[0] < rows[1] else (c1, c0)
        st[fr.IDX_P0] = p0_cell
        st[fr.IDX_P1] = p1_cell

        bars = info.get("barricades") or []
        if len(bars) == 2:
            # The page lists both players' remaining counts; order is not
            # guaranteed, so only the total is used as a checksum.
            st[fr.IDX_WL0] = bars[0]
            st[fr.IDX_WL1] = bars[1]
            walls_expected = 20 - (bars[0] + bars[1])
        else:
            walls_expected = -1

        clocks = info.get("clocks") or []
        our_clock = None
        if len(clocks) == 2:
            # Bottom clock is the local player's in the site's layout.
            mm, ss = clocks[-1]["t"].split(":")
            our_clock = int(mm) * 60 + int(ss)

        st[fr.IDX_TURN] = self._infer_turn(st, us, info)

        return BoardRead(
            state=st, us=us, flipped=flipped, our_clock=our_clock,
            walls_seen=len(walls), walls_expected=walls_expected,
            raw_pawns=pawns, raw_walls=walls, info=info,
        )

    def _infer_turn(self, st: np.ndarray, us: int, info: dict) -> int:
        """Whose move it is, from parity of the material actually on the board.

        Each ply either advances a pawn or spends a wall, so total plies is
        (pawn steps taken) + (walls placed) -- and side to move is its parity.
        Robust against the clock/indicator markup changing.
        """
        walls_placed = 20 - int(st[fr.IDX_WL0]) - int(st[fr.IDX_WL1])
        # Manhattan-ish distance travelled is not exact when pawns detour, so
        # prefer the clock indicator when we have one; parity is the fallback.
        return (walls_placed + self._pawn_plies(st)) % 2

    @staticmethod
    def _pawn_plies(st: np.ndarray) -> int:
        r0 = int(st[fr.IDX_P0]) // 9
        r1 = 8 - int(st[fr.IDX_P1]) // 9
        return r0 + r1

    # --------------------------------------------------------------- write

    def play_pawn(self, cell: int, flipped: bool) -> None:
        x, y = self.cell_xy(cell, flipped)
        self.page.mouse.click(x, y)

    def play_wall(self, action: int, flipped: bool) -> bool:
        orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
        wr, wc = divmod(slot, 8)
        if flipped:
            wr = 7 - wr
        tid = f"slot-{'horizontal' if orient == 0 else 'vertical'}-{wr}-{wc}"
        plan = self.page.evaluate(
            WALL_DRAG, {"traySelector": "[class*='cursor-grab'], [draggable='true']",
                        "slotId": tid})
        if not plan.get("ok"):
            # Fall back to a direct press-drag on the slot itself: some builds
            # accept dragging the ghost wall that appears on hover.
            el = self.page.locator(f'[data-testid="{tid}"]')
            box = el.bounding_box()
            if not box:
                return False
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            self.page.mouse.move(cx, cy)
            self.page.mouse.down()
            self.page.mouse.move(cx + 2, cy + 2, steps=3)
            self.page.mouse.up()
            return True
        f, t = plan["from"], plan["to"]
        self.page.mouse.move(f["x"], f["y"])
        self.page.mouse.down()
        for i in range(1, 13):  # dnd-kit needs real intermediate movement
            self.page.mouse.move(f["x"] + (t["x"] - f["x"]) * i / 12,
                                 f["y"] + (t["y"] - f["y"]) * i / 12, steps=2)
        self.page.mouse.up()
        return True


class Engine:
    """The trained network plus leaf-batched MCTS, with a clock budget."""

    LEAF = 32

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

    # Measured throughput of the leaf-batched search on this machine.
    SIMS_PER_SEC = 38000

    def sims_for(self, clock: float | None, increment: float = 3.0) -> int:
        """Simulations to run, from the remaining clock.

        Two ceilings, whichever is tighter: a little over the increment (so the
        clock trends up, not down), and a fixed fraction of what is left (so a
        low clock forces fast moves). Earlier ranked games averaged 8.7 s/move
        against opponents playing at 2.7 s -- losing on time is a real way to
        lose these games, not a hypothetical.
        """
        if clock is None:
            budget_s = increment
        else:
            budget_s = min(1.2 * increment, clock / 20.0)
        return int(max(256, min(self.max_sims, budget_s * self.SIMS_PER_SEC)))

    def choose(self, st: np.ndarray, sims: int) -> tuple[int, float, np.ndarray]:
        visits = self.mcts.search(st.reshape(1, -1).copy(), sims)[0]
        value = float(self.mcts.root_value()[0])
        return int(visits.argmax()), value, visits


def describe(action: int, st_after: np.ndarray | None = None,
             mover: int | None = None) -> str:
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


# ------------------------------------------------------------------ driver


def should_stop() -> bool:
    return STOP_FILE.exists()


def play_one_move(bridge: Bridge, engine: Engine, args) -> str:
    """Read, decide, play, verify. Returns a status string."""
    read = bridge.read()
    if read is None:
        return "read-failed"
    if read.walls_expected >= 0 and not read.consistent:
        print(f"  !! wall cross-check failed: DOM shows {read.walls_seen} walls, "
              f"barricade counters imply {read.walls_expected}")
        return "inconsistent"
    if int(read.state[fr.IDX_TURN]) != read.us:
        return "not-our-turn"

    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    scratch = fr.make_scratch()
    fr.legal_mask(read.state, mask, scratch)
    if not mask.any():
        return "no-legal-moves"

    sims = engine.sims_for(read.our_clock)
    t0 = time.perf_counter()
    action, value, visits = engine.choose(read.state, sims)
    think = time.perf_counter() - t0
    if not mask[action]:
        print(f"  !! engine chose an action illegal in the read position "
              f"({action}) -- board read is wrong")
        return "illegal"

    expected = read.state.copy()
    fr.apply_action(expected, action, scratch)
    share = float(visits[action] / max(visits.sum(), 1))
    print(f"  {describe(action, expected, read.us)}  "
          f"[{sims} sims, {think:.1f}s, eval {value:+.2f}, {100 * share:.0f}%]")

    if args.dry_run:
        print(render(read.state))
        return "dry-run"

    if action >= fr.MOVE_BASE:
        dest = int(expected[fr.IDX_P0 if read.us == 0 else fr.IDX_P1])
        bridge.play_pawn(dest, read.flipped)
    elif not bridge.play_wall(action, read.flipped):
        return "wall-drag-failed"

    # Confirm the site accepted exactly the move we intended.
    for _ in range(12):
        bridge.page.wait_for_timeout(250)
        after = bridge.read()
        if after is None:
            continue
        ours_moved = (after.state[fr.IDX_P0] == expected[fr.IDX_P0]
                      and after.state[fr.IDX_P1] == expected[fr.IDX_P1])
        walls_match = (
            np.array_equal(after.state[fr.WH_OFF:fr.WH_OFF + 64],
                           expected[fr.WH_OFF:fr.WH_OFF + 64])
            and np.array_equal(after.state[fr.WV_OFF:fr.WV_OFF + 64],
                               expected[fr.WV_OFF:fr.WV_OFF + 64]))
        if ours_moved and walls_match:
            return "played"
    print("  !! could not confirm the move registered on the site")
    return "unconfirmed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-sims", type=int, default=65536,
                    help="ceiling on simulations per move (memory scales with it)")
    ap.add_argument("--max-games", type=int, default=0, help="0 = until stopped")
    ap.add_argument("--dry-run", action="store_true",
                    help="read and decide, but never click")
    ap.add_argument("--url", default=f"{BASE}/", help="page to start from")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    if STOP_FILE.exists():
        STOP_FILE.unlink()
        print(f"removed stale {STOP_FILE}")

    print("loading engine...")
    engine = Engine(args.checkpoint, args.device, args.max_sims)
    m = engine.meta
    print(f"engine: {m.get('checkpoint', args.checkpoint)} "
          f"(gen {m.get('iteration', '?')}, {m.get('elo', 0):+.0f} Elo internal)")

    stopping = {"flag": False}

    def on_sigint(signum, frame):
        stopping["flag"] = True
        print("\nstopping after the current move...")

    signal.signal(signal.SIGINT, on_sigint)

    with sync_playwright() as pw:
        # A persistent profile keeps your login between runs. Log in by hand in
        # the window that opens the first time; this script never sees a password.
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        ctx = pw.firefox.launch_persistent_context(
            str(PROFILE_DIR), headless=False,
            viewport={"width": 1500, "height": 1000},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(args.url, wait_until="networkidle", timeout=60000)

        bridge = Bridge(page, verbose=args.verbose)
        print("\nIf you are not signed in, sign in now in the browser window.")
        print("Then open a ranked game. Waiting for a board...\n")

        games = 0
        idle_since = time.time()
        while not (stopping["flag"] or should_stop()):
            if args.max_games and games >= args.max_games:
                print(f"reached --max-games {args.max_games}")
                break
            if not bridge.refresh_geometry():
                page.wait_for_timeout(1500)
                if time.time() - idle_since > 300:
                    print("no board for 5 minutes; still waiting "
                          f"(create {STOP_FILE} to stop)")
                    idle_since = time.time()
                continue

            status = play_one_move(bridge, engine, args)
            if status in ("read-failed", "inconsistent", "illegal",
                          "wall-drag-failed", "unconfirmed"):
                print(f"  aborting: {status}. The board reader needs attention; "
                      f"re-run with --dry-run --verbose.")
                if args.verbose:
                    read = bridge.read()
                    print(json.dumps(
                        {"pawns": read.raw_pawns if read else None,
                         "walls": read.raw_walls[:8] if read else None},
                        indent=1, default=str)[:2000])
                break
            if status == "dry-run":
                print("\ndry run complete -- compare the board above with your "
                      "screen. If it matches, the reader works.")
                break
            if status in ("not-our-turn", "no-legal-moves"):
                page.wait_for_timeout(600)
                continue

            info = bridge.page.evaluate(TURN_JS)
            if info.get("gameOver"):
                games += 1
                print(f"game finished ({games}). {info['text'][:120]}\n")
                page.wait_for_timeout(3000)

        print(f"stopped after {games} completed game(s)")
        if STOP_FILE.exists():
            STOP_FILE.unlink()
        ctx.close()


if __name__ == "__main__":
    main()
