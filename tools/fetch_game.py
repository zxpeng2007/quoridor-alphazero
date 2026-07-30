"""Fetch a barricade.gg game record with ground-truth wall positions.

Loads the analysis page headless, sniffs any JSON the page fetches, steps the
replay to the final move, and dumps the occupied wall slots (by testid) plus
pawn positions. Matching the slot list against the move list pins down the
site's wall-notation convention exactly, with no guessing.
"""

from __future__ import annotations

import json
import sys

from playwright.sync_api import sync_playwright

GAME = sys.argv[1] if len(sys.argv) > 1 else "b8ewfg"
CAPTURED: list[dict] = []

JS_FINAL = """() => {
  const s00 = document.querySelector('[data-testid="slot-horizontal-0-0"]');
  if (!s00) return {found: false};
  const board = s00.parentElement;
  const b = board.getBoundingClientRect();
  const walls = [], squares = [];
  for (const el of board.querySelectorAll('[data-testid]')) {
    const st = getComputedStyle(el);
    const kid = el.firstElementChild ? getComputedStyle(el.firstElementChild) : null;
    const occupied =
      st.backgroundColor !== 'rgba(0, 0, 0, 0)' ||
      (kid && kid.backgroundColor !== 'rgba(0, 0, 0, 0)' && kid.opacity !== '0');
    if (occupied) walls.push({tid: el.dataset.testid,
                              bg: st.backgroundColor,
                              kidBg: kid ? kid.backgroundColor : null});
  }
  for (const el of board.children) {
    const r = el.getBoundingClientRect();
    if (r.width > 40 && r.width < 75 && Math.abs(r.width - r.height) < 6) {
      const st = getComputedStyle(el);
      const kid = el.firstElementChild;
      if (kid && kid.getBoundingClientRect().width > 20) {
        squares.push({
          x: Math.round(r.x - b.x), y: Math.round(r.y - b.y),
          kidCls: (kid.className || '').toString().slice(0, 80),
          kidBg: getComputedStyle(kid).backgroundColor,
        });
      }
    }
  }
  return {found: true, boardX: b.x, boardY: b.y, walls, pawnish: squares};
}"""


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        def on_response(resp):
            ct = resp.headers.get("content-type", "")
            if "json" in ct and len(CAPTURED) < 30:
                try:
                    body = resp.json()
                    CAPTURED.append({"url": resp.url[:140], "body": body})
                except Exception:
                    pass

        page.on("response", on_response)
        page.goto(f"https://barricade.gg/analysis?game={GAME}",
                  wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(1500)

        print("==== captured JSON responses ====")
        for c in CAPTURED:
            print(f"\n-- {c['url']}")
            print(json.dumps(c["body"], default=str)[:2600])

        # Step to the end of the game and read the final board.
        last = page.get_by_role("button", name="Go to last move")
        if last.count():
            last.first.click()
            page.wait_for_timeout(1500)
        print("\n==== final board ====")
        print(json.dumps(page.evaluate(JS_FINAL), indent=1)[:5000])
        print("\n==== page text ====")
        print(page.inner_text("body")[:900])
        browser.close()


if __name__ == "__main__":
    main()
