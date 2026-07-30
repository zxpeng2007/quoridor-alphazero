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

  // Orientation from the rank labels. Demand a full, strictly monotonic run:
  // reading a partially rendered board could otherwise yield two labels in an
  // arbitrary order, and a wrong flip maps our own pawn onto its goal row,
  // making a fresh game look already won.
  const labels = [];
  board.querySelectorAll('div').forEach(el => {
    const t = el.textContent.trim();
    if (/^[1-9]$/.test(t) && el.children.length === 0)
      labels.push({r: +t, y: el.getBoundingClientRect().y});
  });
  labels.sort((a, b) => a.y - b.y);
  let flipped = null;
  if (labels.length === 9) {
    const ranks = labels.map(l => l.r);
    const desc = ranks.every((v, i) => i === 0 || ranks[i - 1] === v + 1);
    const asc = ranks.every((v, i) => i === 0 || ranks[i - 1] === v - 1);
    if (desc) flipped = true;         // rank 9 at the top: the default view
    else if (asc) flipped = false;    // rotated for the second seat
  }

  const pawns = [];
  board.querySelectorAll('div.rounded-full').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width < 10) return;
    const cls = (el.className || '').toString();
    const colour = cls.includes('bg-red') ? 'red'
                 : cls.includes('bg-blue') ? 'blue' : null;
    if (!colour) return;
    // Keep the fractional grid position: the site animates pawn movement, and
    // a pawn caught between squares must be rejected, not rounded to a
    // plausible-looking wrong cell.
    pawns.push({colour,
                colF: (r.x + r.width / 2 - ox - cell / 2) / pitch,
                vrowF: (r.y + r.height / 2 - oy - cell / 2) / pitch});
  });

  // Distinguishing a placed wall from the hover preview cannot be done by
  // colour: live games paint walls grey (#cccccc), the same family as the
  // gray-300 preview, and the analysis view paints them red/blue. The reliable
  // difference is structural -- the preview is defined by `group-hover:`
  // classes, a real wall is not.
  const walls = [];
  document.querySelectorAll('[data-testid^="slot-"]').forEach(el => {
    for (const kid of el.querySelectorAll('div')) {
      const cls = (kid.className || '').toString();
      if (cls.includes('group-hover')) continue;          // hover preview
      const ks = getComputedStyle(kid);
      if (ks.backgroundColor === 'rgba(0, 0, 0, 0)' &&
          ks.backgroundImage === 'none') continue;
      if (parseFloat(ks.opacity) < 0.9) continue;
      walls.push({tid: el.dataset.testid, bg: ks.backgroundColor});
      break;
    }
  });

  // The site rings every legal pawn destination while it is your move, and
  // shows none otherwise. That is both the turn signal and an independent
  // check on the decoded position: if the site's legal destinations disagree
  // with the engine's, the board has been misread.
  const highlights = [];
  for (const el of board.children) {
    if (el.dataset.testid) continue;
    const cls = (el.className || '').toString();
    if (!cls.includes('ring-2') && !cls.includes('animate-pulse')) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 30) continue;
    // Carry the real screen centre too: clicking the element the site itself
    // rings is exact, where recomputing the square from board geometry is not.
    highlights.push({col: Math.round((r.x + r.width / 2 - ox - cell / 2) / pitch),
                     vrow: Math.round((r.y + r.height / 2 - oy - cell / 2) / pitch),
                     cx: r.x + r.width / 2, cy: r.y + r.height / 2});
  }

  const text = document.body.innerText;

  // Game over requires a *visible* result element, not a text match anywhere
  // on the page. Matching page text caught a permanently-mounted, hidden
  // "Play again anytime" promo and declared every fresh board finished.
  // Interactive controls are never a result. A ranked game offers a "Draw"
  // button mid-game, and treating that as a result made the bridge abandon
  // games in progress. Only a non-interactive banner counts, plus the one
  // button that exists solely after a game: Rematch.
  let gameOverLabel = null;
  document.querySelectorAll('div,button,span,h1,h2,h3,p').forEach(el => {
    if (gameOverLabel !== null) return;
    if (el.children.length > 1) return;
    const s = (el.textContent || '').trim();
    if (!s || s.length > 30) return;
    const interactive = el.tagName === 'BUTTON' || el.tagName === 'A' ||
                        el.getAttribute('role') === 'button' ||
                        el.closest('button,a,[role="button"]') !== null;
    const isResult = interactive
      ? /^rematch$/i.test(s)
      : (/^(victory|defeat)\b/i.test(s) ||
         /^you (won|lost|win|lose)\b/i.test(s) ||
         /^(it'?s a |game )?draw[!.]?$/i.test(s));
    if (!isResult) return;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) gameOverLabel = s;
  });

  // Each player's remaining barricades, taken from its own panel. The panel
  // carries a colour swatch (bg-red-500 / bg-blue-500) matching that player's
  // pawn, which identifies the owner outright. Ownership cannot be read from
  // the walls themselves -- live games paint every wall the same grey.
  const bars = [];
  document.querySelectorAll('div').forEach(el => {
    const m = (el.textContent || '').trim()
      .match(/^Barricades:\s*(\d+)\s*\/\s*10$/);
    if (!m) return;
    const swatch = el.querySelector('span[class*="bg-"]');
    const scls = swatch ? (swatch.className || '').toString() : '';
    const colour = scls.includes('bg-red') ? 'red'
                 : scls.includes('bg-blue') ? 'blue' : null;
    bars.push({n: +m[1], colour, y: el.getBoundingClientRect().y});
  });
  bars.sort((a, b) => a.y - b.y);
  return {geo: {bx: bb.x, by: bb.y, pitch, cell, gap, ox, oy},
          flipped, pawns, walls, highlights, barricades: bars,
          youArePlayer: (text.match(/You are Player (\d)/) || [])[1] || null,
          gameOver: gameOverLabel !== null, gameOverLabel,
          text: text.slice(0, 300)};
}"""

TRAY_JS = r"""() => {
  // The two draggable barricade pieces; the wide one places horizontal walls.
  // Identify them by the grab cursor rather than by role, and require they be
  // visible -- the tray lives in a collapsible panel that some views start
  // closed, in which case the pieces are absent or zero-sized.
  const out = [];
  document.querySelectorAll('div,button,span').forEach(el => {
    const st = getComputedStyle(el);
    const cls = (el.className || '').toString();
    const grabby = st.cursor === 'grab' || st.cursor === 'grabbing' ||
                   cls.includes('cursor-grab');
    if (!grabby) return;
    if (st.visibility === 'hidden' || st.display === 'none') return;
    const r = el.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;
    if (Math.abs(r.width - r.height) < 4) return;   // need a clear orientation
    out.push({horizontal: r.width > r.height,
              cx: r.x + r.width / 2, cy: r.y + r.height / 2,
              w: Math.round(r.width), h: Math.round(r.height)});
  });
  // Prefer the outermost container per orientation (largest of each).
  const pick = (horiz) => out.filter(t => t.horizontal === horiz)
                             .sort((a, b) => (b.w * b.h) - (a.w * a.h))[0];
  return [pick(true), pick(false)].filter(Boolean);
}"""

TRAY_TOGGLE_JS = r"""() => {
  // The panel header that expands the barricade tray, if it is collapsed.
  const btns = [...document.querySelectorAll('button,[role="button"]')];
  const hit = btns.find(b => /barricade/i.test((b.textContent || '').trim())
                             && (b.textContent || '').trim().length < 60);
  if (!hit) return null;
  const r = hit.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return null;
  return {cx: r.x + r.width / 2, cy: r.y + r.height / 2,
          text: hit.textContent.trim().slice(0, 40)};
}"""

LOBBY_READY_JS = r"""() => {
  // The lobby is identifiable by its Casual/Ranked mode toggle.
  const b = [...document.querySelectorAll('button')].map(x => x.textContent.trim());
  return b.includes('Ranked') && b.includes('Casual');
}"""

RANKED_SELECTED_JS = r"""() => {
  const b = [...document.querySelectorAll('button')]
    .find(x => x.textContent.trim() === 'Ranked');
  if (!b) return false;
  // The active tab in the segmented control is the filled one.
  const cls = (b.className || '').toString();
  return /bg-amber|bg-white|text-white/.test(cls);
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
    # Legal pawn destinations per the site: engine cell -> screen centre.
    highlights: dict[int, tuple[float, float]]
    raw: dict = field(repr=False)

    @property
    def our_turn(self) -> bool:
        # The site only rings destinations on the side to move.
        return bool(self.highlights)


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
            # Transient while the board is still rendering; wait rather than
            # guess, since a wrong orientation mirrors the whole position.
            return None, "wait: board orientation not readable yet"
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
            col_f, vrow_f = float(p["colF"]), float(p["vrowF"])
            col, vrow = int(round(col_f)), int(round(vrow_f))
            # A pawn mid-slide sits between squares. Rounding it produces a
            # legal-looking position for the wrong board, so wait instead.
            if abs(col_f - col) > 0.2 or abs(vrow_f - vrow) > 0.2:
                return None, "wait: a pawn is between squares (animating)"
            row = 8 - vrow if flipped else vrow
            if not (0 <= row < 9 and 0 <= col < 9):
                return None, f"pawn off board: {p}"
            by_colour[p["colour"]] = row * 9 + col
        if set(by_colour) != {"red", "blue"}:
            return None, (f"expected one red and one blue pawn, "
                          f"found {dom['pawns']}")
        st[fr.IDX_P0] = by_colour["red"]
        st[fr.IDX_P1] = by_colour["blue"]

        you = dom.get("youArePlayer")
        us = 0 if you == "1" else 1 if you == "2" else None
        if us is None:
            return None, "could not determine which seat we are playing"

        bars = dom.get("barricades") or []
        walls_expected = -1
        if len(bars) == 2:
            walls_expected = 20 - (bars[0]["n"] + bars[1]["n"])
            by_col = {b["colour"]: b["n"] for b in bars if b["colour"]}
            if {"red", "blue"} <= set(by_col):
                # Same convention as the pawns: red is the first seat.
                st[fr.IDX_WL0] = by_col["red"]
                st[fr.IDX_WL1] = by_col["blue"]
            else:
                # Fallback: panels run opponent-above-us, since the board
                # always orients the local player at the bottom.
                st[fr.IDX_WL0 if us == 0 else fr.IDX_WL1] = bars[1]["n"]
                st[fr.IDX_WL0 if us == 1 else fr.IDX_WL1] = bars[0]["n"]
        else:
            return None, (f"expected two barricade counters, found {len(bars)} "
                          "-- cannot tell how many walls remain")

        highlights: dict[int, tuple[float, float]] = {}
        for h in dom.get("highlights") or []:
            vrow, col = int(round(h["vrow"])), int(round(h["col"]))
            row = 8 - vrow if flipped else vrow
            if 0 <= row < 9 and 0 <= col < 9:
                highlights[row * 9 + col] = (h["cx"], h["cy"])

        return BoardRead(state=st, flipped=flipped, us=us,
                         walls_seen=len(dom["walls"]),
                         walls_expected=walls_expected,
                         game_over=bool(dom.get("gameOver")),
                         highlights=highlights, raw=dom), ""

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

    def play_pawn(self, cell: int, read: "BoardRead") -> bool:
        """Click the destination square the site itself has ringed.

        Using the highlight element's own centre removes any dependence on
        recomputing the square from board geometry.
        """
        target = read.highlights.get(cell)
        if target is None:
            print(f"  !! destination cell {cell} is not among the site's "
                  f"highlighted moves {sorted(read.highlights)}")
            return False
        self.page.mouse.click(target[0], target[1])
        return True

    def _click_text(self, patterns: tuple[str, ...], timeout: float = 4000) -> str | None:
        """Click the first visible button/link whose text matches any pattern."""
        for pat in patterns:
            for role in ("button", "link"):
                loc = self.page.get_by_role(role, name=re.compile(pat, re.I))
                try:
                    if loc.count():
                        loc.first.click(timeout=timeout)
                        return pat
                except Exception:
                    continue
        return None

    def back_to_lobby_and_queue(self, wait_s: float = 120.0) -> bool:
        """Leave a finished game and queue for the next ranked match.

        Returns True once a board is on screen again.
        """
        self._click_text((r"back to lobby", r"^lobby$"))
        # Confirm we actually reached the lobby rather than assuming the click
        # landed; otherwise the matchmaking button is looked for on a game page.
        for _ in range(12):
            self.page.wait_for_timeout(500)
            if self.page.evaluate(LOBBY_READY_JS):
                break
        else:
            self.page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=45000)
            self.page.wait_for_timeout(2000)
        if not self.page.evaluate(LOBBY_READY_JS):
            print("  !! could not reach the lobby")
            return False

        # Select the ranked tab and verify it took, so a stray state cannot
        # quietly queue a casual game.
        self._click_text((r"^ranked$",))
        self.page.wait_for_timeout(500)
        if not self.page.evaluate(RANKED_SELECTED_JS):
            print("  !! could not select the Ranked tab")
            return False

        started = self._click_text((r"find ranked match", r"find match",
                                    r"^play now$"))
        if started is None:
            print("  !! could not find the ranked matchmaking button; "
                  "start a game manually and it will be picked up")
            return False
        print(f"  queued for a ranked match ({started!r}); waiting for pairing...")

        deadline = time.time() + wait_s
        while time.time() < deadline:
            if STOP_FILE.exists():
                return False
            try:
                self.page.wait_for_selector('[data-testid="slot-horizontal-0-0"]',
                                            timeout=2000)
                self.page.wait_for_timeout(1200)   # let the board settle
                return True
            except Exception:
                continue
        print("  !! no match found within the wait window")
        return False

    def _tray_piece(self, want_horizontal: bool) -> dict | None:
        """Locate a draggable barricade piece, expanding the tray if collapsed."""
        for attempt in range(2):
            hits = [t for t in self.page.evaluate(TRAY_JS)
                    if t["horizontal"] == want_horizontal]
            if hits:
                return hits[0]
            if attempt == 0:
                toggle = self.page.evaluate(TRAY_TOGGLE_JS)
                if toggle is None:
                    break
                print(f"  (expanding barricade tray: {toggle['text']!r})")
                self.page.mouse.click(toggle["cx"], toggle["cy"])
                self.page.wait_for_timeout(700)
        print("  !! could not find the barricade tray piece to drag "
              f"({'horizontal' if want_horizontal else 'vertical'}). "
              "Open the 'Drag & Drop Barricades' panel in the browser and retry.")
        return None

    def play_wall(self, action: int, flipped: bool) -> bool:
        """Place a wall by dragging the matching tray piece onto the slot.

        Clicking a slot does NOT work: a bare click always commits the tray's
        currently selected orientation, which is vertical, so clicking a
        horizontal slot silently places a vertical wall at the same coordinates
        (verified against the live site). Selecting the tray piece first does
        not change that. Only the drag carries the orientation.
        """
        orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
        wr, wc = divmod(slot, 8)
        if flipped:
            wr = 7 - wr
        want_horizontal = orient == 0
        tid = f"slot-{'horizontal' if want_horizontal else 'vertical'}-{wr}-{wc}"

        el = self.page.locator(f'[data-testid="{tid}"]')
        if not el.count():
            return False
        box = el.first.bounding_box()
        if not box:
            return False

        src = self._tray_piece(want_horizontal)
        if src is None:
            return False
        tx, ty = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

        # dnd-kit ignores a teleporting cursor; it needs real intermediate moves.
        self.page.mouse.move(src["cx"], src["cy"])
        self.page.mouse.down()
        for i in range(1, 21):
            self.page.mouse.move(src["cx"] + (tx - src["cx"]) * i / 20,
                                 src["cy"] + (ty - src["cy"]) * i / 20)
            self.page.wait_for_timeout(20)
        self.page.wait_for_timeout(250)
        self.page.mouse.up()
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
        """Returns ``(action, value, visits)``; action is -1 if there is none.

        A search from an already-decided position returns zero visits
        everywhere, and argmax of an all-zero array is 0 -- which happens to be
        a wall placement. Report "no move" instead of inventing one.
        """
        visits = self.mcts.search(st.reshape(1, -1).copy(), sims)[0]
        if visits.sum() <= 0:
            return -1, 0.0, visits
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
        if err.startswith("wait:"):
            return "no-board"          # still rendering; poll again
        print(f"  !! board read failed: {err}")
        return "read-failed"
    banner = read.raw.get("gameOverLabel")
    won = fr.winner(read.state)
    if won >= 0:
        # A pawn on its goal row is decisive. Orientation is only accepted from
        # a complete, monotonic set of rank labels, so this cannot come from a
        # mirrored read; stale highlights sometimes linger after the win and
        # must not veto it.
        if args.verbose:
            print(f"  game-over: {'we' if won == read.us else 'opponent'} "
                  f"reached the goal row")
        return "game-over"
    if read.game_over:
        # A banner with no winner means resignation, timeout, or a stray match.
        # Here the highlights are worth trusting: the site rings destinations
        # only for the side to move, so their presence means the game is live.
        if read.our_turn:
            print(f"  ignoring a game-over banner ({banner!r}): it is our turn "
                  f"with {len(read.highlights)} legal moves, so the game is live")
            if args.verbose:
                print(render(read.state))
        else:
            if args.verbose:
                print(f"  game-over: banner {banner!r}")
            return "game-over"
    if read.walls_expected >= 0 and read.walls_seen != read.walls_expected:
        # A wall placed moments ago may still be fading in (the slot child
        # carries a 150ms transition), so the DOM can legitimately lag the
        # counters for an instant. Re-read a few times; only a persistent
        # disagreement means we have actually misread the board.
        settled = None
        for _ in range(5):
            bridge.page.wait_for_timeout(300)
            dom_retry = bridge.snapshot()
            cand, _ = bridge.decode(dom_retry) if dom_retry else (None, "")
            if cand is not None and cand.walls_seen == cand.walls_expected:
                settled = cand
                break
        if settled is None:
            print(f"  !! wall cross-check failed: {read.walls_seen} walls in the "
                  f"DOM, barricade counters imply {read.walls_expected}")
            return "inconsistent"
        read = settled
        if read.game_over:
            return "game-over"

    # A dry run exists to validate the *reader*, so it reports the position
    # unconditionally rather than waiting for our turn -- otherwise a turn
    # detection problem hides the board decode entirely.
    read.state[fr.IDX_TURN] = read.us
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    dests = np.zeros(12, dtype=np.int32)
    fr.legal_mask_dests(read.state, mask, scratch, dests)
    engine_dests = {int(dests[i]) for i in range(12)
                    if mask[fr.MOVE_BASE + i]}

    if args.dry_run:
        print(f"  seat: engine player {read.us} "
              f"({'red' if read.us == 0 else 'blue'}), "
              f"board {'flipped' if read.flipped else 'normal'}")
        print(f"  walls: {read.walls_seen} on board, counters imply "
              f"{read.walls_expected}")
        print(f"  turn: {'ours' if read.our_turn else 'opponent (no highlights)'}"
              f"  [{len(read.highlights)} highlighted destinations]")
        print(render(read.state))
        if read.our_turn:
            agree = set(read.highlights) == engine_dests
            print(f"  legality cross-check: site {sorted(read.highlights)} vs "
                  f"engine {sorted(engine_dests)} -> "
                  f"{'AGREE' if agree else 'MISMATCH (board misread)'}")
            if agree and mask.any():
                action, value, _ = engine.choose(read.state, 4096)
                after = read.state.copy()
                fr.apply_action(after, action, scratch)
                print(f"  engine would play: "
                      f"{describe(action, after, read.us)} (eval {value:+.2f})")
        return "dry-run"

    if not read.our_turn:
        return "not-our-turn"
    if not mask.any():
        return "no-legal-moves"

    # Independent confirmation that we decoded the right position: the site's
    # own legal destinations must match the engine's. A disagreement is usually
    # transient (something still settling on the page), so re-read before
    # treating it as a real misread.
    if set(read.highlights) != engine_dests:
        settled = None
        for _ in range(5):
            bridge.page.wait_for_timeout(300)
            dom_r = bridge.snapshot()
            cand, _err = bridge.decode(dom_r) if dom_r else (None, "")
            if cand is None or not cand.our_turn:
                continue
            cand.state[fr.IDX_TURN] = cand.us
            fr.legal_mask_dests(cand.state, mask, scratch, dests)
            cand_dests = {int(dests[i]) for i in range(12)
                          if mask[fr.MOVE_BASE + i]}
            if set(cand.highlights) == cand_dests:
                settled, engine_dests = cand, cand_dests
                break
        if settled is None:
            print(f"  !! legality mismatch: site offers {sorted(read.highlights)}, "
                  f"engine computes {sorted(engine_dests)}")
            print(f"    decoded P0={int(read.state[fr.IDX_P0])} "
                  f"P1={int(read.state[fr.IDX_P1])} flipped={read.flipped} "
                  f"us={read.us} walls={read.walls_seen}/{read.walls_expected}")
            print(render(read.state))
            return "legality-mismatch"
        read = settled

    clocks = bridge.page.evaluate(CLOCKS_JS)
    our_clock = clock_seconds(clocks[1]["t"]) if len(clocks) == 2 else None
    sims = engine.sims_for(our_clock)

    t0 = time.perf_counter()
    action, value, visits = engine.choose(read.state, sims)
    think = time.perf_counter() - t0
    if action < 0:
        return "no-move"
    if not mask[action]:
        print("  !! engine chose an action illegal in the read position")
        return "illegal"

    expected = read.state.copy()
    fr.apply_action(expected, action, scratch)
    share = float(visits[action] / max(visits.sum(), 1))
    print(f"  {describe(action, expected, read.us):<16} "
          f"[{sims} sims, {think:.1f}s, eval {value:+.2f}, {100 * share:.0f}%"
          + (f", clock {our_clock:.0f}s" if our_clock else "") + "]")

    our_idx = fr.IDX_P0 if read.us == 0 else fr.IDX_P1
    if action >= fr.MOVE_BASE:
        if not bridge.play_pawn(int(expected[our_idx]), read):
            return "pawn-click-failed"
    elif not bridge.play_wall(action, read.flipped):
        return "wall-place-failed"

    # Confirm only that OUR move landed. The opponent usually replies within a
    # second, so requiring the whole board to equal the position after our move
    # alone would essentially never hold.
    want_wall = None
    if action < fr.MOVE_BASE:
        orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
        want_wall = (fr.WH_OFF if orient == 0 else fr.WV_OFF) + slot

    for _ in range(16):
        bridge.page.wait_for_timeout(250)
        dom2 = bridge.snapshot()
        after, _ = bridge.decode(dom2) if dom2 else (None, "")
        if after is None:
            continue
        if want_wall is None:
            if after.state[our_idx] == expected[our_idx]:
                return "played"
        elif after.state[want_wall]:
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
    ap.add_argument("--no-requeue", dest="requeue", action="store_false",
                    help="do not auto-queue the next ranked game after a result")
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
        last_note = 0.0
        last_status = ""
        # A finished board stays on screen until the next game starts, so the
        # game-over signal repeats. Count a result once, then require a live
        # board again before another can be counted.
        result_counted = False
        while not (stopping["flag"] or STOP_FILE.exists()):
            if args.max_games and games >= args.max_games:
                print(f"reached --max-games {args.max_games}")
                break
            status = play_one_move(bridge, engine, args)

            # Never sit silent: say what we are waiting for, at most every 5s.
            if status in ("no-board", "not-our-turn", "turn-unknown",
                          "no-legal-moves", "no-move"):
                now = time.time()
                if now - last_note > 5.0 or status != last_status:
                    reason = {
                        "no-board": "no board on the page (open a game)",
                        "not-our-turn": "opponent to move (no highlighted cells)",
                        "turn-unknown": "cannot tell whose turn it is",
                        "no-legal-moves": "no legal moves in the read position",
                        "no-move": "position is already decided",
                    }[status]
                    print(f"  waiting: {reason}")
                    last_note, last_status = now, status

            if status in ("read-failed", "inconsistent", "illegal",
                          "legality-mismatch", "wall-place-failed",
                          "pawn-click-failed", "unconfirmed"):
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
                if not result_counted:
                    games += 1
                    result_counted = True
                    print(f"game finished ({games})")
                if args.max_games and games >= args.max_games:
                    continue          # loop head will stop us
                if args.requeue:
                    page.wait_for_timeout(1500)
                    if bridge.back_to_lobby_and_queue():
                        print("  new game started\n")
                        result_counted = False
                    else:
                        page.wait_for_timeout(3000)
                else:
                    page.wait_for_timeout(2500)
                continue
            if status in ("no-board", "not-our-turn", "turn-unknown",
                          "no-legal-moves", "no-move"):
                page.wait_for_timeout(700)
                continue
            if status == "played":
                result_counted = False   # a live board again

        print(f"stopped after {games} completed game(s)")
        if STOP_FILE.exists():
            STOP_FILE.unlink()
        ctx.close()


if __name__ == "__main__":
    main()
