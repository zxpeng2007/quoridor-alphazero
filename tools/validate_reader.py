"""Validate the DOM board reader against engine ground truth.

Steps through a finished game on the analysis page and, at every ply, compares
what the reader decodes from the DOM against the position obtained by replaying
the game's own move list through the rules engine. Any mismatch -- a pawn on the
wrong square, a missing wall, a mirrored row -- is reported with the ply number.

This is the check that makes ``tools/autoplay.py`` trustworthy: a misread board
does not produce an obviously bad move, it produces a legal move for the wrong
position, which looks fine and loses.

    python tools/validate_reader.py 0jm2n3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np
from playwright.sync_api import sync_playwright

from quoridor import fastrules as fr

_spec = importlib.util.spec_from_file_location("ag", "tools/analyze_game.py")
ag = importlib.util.module_from_spec(_spec)
sys.modules["ag"] = ag
_spec.loader.exec_module(ag)

# Kept identical to the copy in autoplay.py -- this is the contract under test.
BOARD_JS = r"""() => {
  const s00 = document.querySelector('[data-testid="slot-horizontal-0-0"]');
  const s11 = document.querySelector('[data-testid="slot-horizontal-1-1"]');
  if (!s00 || !s11) return null;
  const board = s00.parentElement;
  const bb = board.getBoundingClientRect();
  const r00 = s00.getBoundingClientRect(), r11 = s11.getBoundingClientRect();
  const pitch = r11.x - r00.x, gap = r00.height, cell = pitch - gap;
  const ox = r00.x, oy = r00.y - cell;

  // Orientation from the rank labels: never assume which way the board is drawn.
  const labels = [];
  board.querySelectorAll('div').forEach(el => {
    const t = el.textContent.trim();
    if (/^[1-9]$/.test(t) && el.children.length === 0)
      labels.push({r: +t, y: el.getBoundingClientRect().y});
  });
  labels.sort((a, b) => a.y - b.y);
  const flipped = labels.length >= 2
    ? labels[0].r > labels[labels.length - 1].r : null;

  // Pawns are colour-coded: red is the first seat, blue the second. Colour is
  // authoritative -- position is not, because the pawns cross during a game.
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

  // A placed wall is opaque and coloured; the hover ghost is translucent grey.
  // Slots with no child at all are ones a crossing wall has made unavailable.
  const isWallPaint = (ks) => {
    if (ks.opacity !== '' && parseFloat(ks.opacity) < 0.9) return false;
    const m = ks.backgroundColor.match(/rgba?\(([^)]+)\)/);
    if (!m) return false;
    const p = m[1].split(',').map(s => parseFloat(s));
    const [r, g, b] = p;
    const alpha = p.length > 3 ? p[3] : 1;
    if (alpha < 0.9) return false;
    if (Math.max(r, g, b) - Math.min(r, g, b) < 20) return false;
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
          youArePlayer: (text.match(/You are Player (\d)/) || [])[1] || null};
}"""


def decode(dom: dict) -> tuple[np.ndarray | None, str]:
    """DOM snapshot -> engine state. Returns ``(state, error)``."""
    if dom is None:
        return None, "no board"
    if dom["flipped"] is None:
        return None, "could not read orientation"
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

    by_colour = {}
    for p in dom["pawns"]:
        row = 8 - p["vrow"] if flipped else p["vrow"]
        if not (0 <= row < 9 and 0 <= p["col"] < 9):
            return None, f"pawn off board: {p}"
        by_colour[p["colour"]] = row * 9 + p["col"]
    if set(by_colour) != {"red", "blue"}:
        return None, f"expected one red and one blue pawn, got {list(by_colour)}"

    st[fr.IDX_P0] = by_colour["red"]   # red = first seat = engine player 0
    st[fr.IDX_P1] = by_colour["blue"]
    return st, ""


def compare(got: np.ndarray, want: np.ndarray) -> list[str]:
    diffs = []
    if int(got[fr.IDX_P0]) != int(want[fr.IDX_P0]):
        diffs.append(f"P0 cell {int(got[fr.IDX_P0])} != {int(want[fr.IDX_P0])}")
    if int(got[fr.IDX_P1]) != int(want[fr.IDX_P1]):
        diffs.append(f"P1 cell {int(got[fr.IDX_P1])} != {int(want[fr.IDX_P1])}")
    for name, off in (("h", fr.WH_OFF), ("v", fr.WV_OFF)):
        g = set(np.nonzero(got[off:off + 64])[0].tolist())
        w = set(np.nonzero(want[off:off + 64])[0].tolist())
        if g != w:
            miss = [(i // 8, i % 8) for i in sorted(w - g)]
            extra = [(i // 8, i % 8) for i in sorted(g - w)]
            if miss:
                diffs.append(f"{name}-walls missing {miss}")
            if extra:
                diffs.append(f"{name}-walls spurious {extra}")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game", nargs="?", default="0jm2n3")
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    rec = ag.fetch(args.game)
    tokens = rec["historyCsv"].split(",")
    actions, states, label = ag.decode_game(tokens)
    print(f"{rec['player1Username']} vs {rec['player2Username']}, "
          f"{len(tokens)} moves, decoded as {label}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(f"https://barricade.gg/analysis?game={args.game}",
                  wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(1200)

        # The analysis page opens at the final position; rewind to the start.
        try:
            page.locator('button[title*="first move"]').first.click(timeout=5000)
            page.wait_for_timeout(600)
        except Exception:
            print("  warning: could not rewind to the first move")

        nxt = page.locator('button[title*="next move"]')
        checked = failures = 0
        for ply in range(len(states)):
            dom = page.evaluate(BOARD_JS)
            got, err = decode(dom)
            if got is None:
                print(f"  ply {ply}: READ FAILED - {err}")
                failures += 1
            else:
                diffs = compare(got, states[ply])
                checked += 1
                if diffs:
                    failures += 1
                    print(f"  ply {ply}: MISMATCH - {'; '.join(diffs)}")
            if ply < len(states) - 1:
                try:
                    nxt.first.click(timeout=4000)
                    page.wait_for_timeout(220)
                except Exception:
                    print(f"  stopped stepping at ply {ply}")
                    break
        browser.close()

    print(f"\n{checked} positions compared, {failures} mismatch(es)")
    if failures == 0:
        print("reader agrees with the engine on every position")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
