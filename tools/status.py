"""Summarise a training run from its history file.

Read this rather than the training log when checking on a long run: it makes
promotion rate and loss trend visible at a glance, which are the two things that
actually indicate whether the run is healthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bar(value: float, lo: float, hi: float, width: int = 18) -> str:
    if hi <= lo:
        return " " * width
    frac = min(max((value - lo) / (hi - lo), 0.0), 1.0)
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints")
    ap.add_argument("--last", type=int, default=25, help="iterations to show")
    args = ap.parse_args()

    path = Path(args.dir) / "history.json"
    if not path.exists():
        raise SystemExit(f"no history at {path} -- has training started?")
    history = json.loads(path.read_text())
    if not history:
        raise SystemExit("history is empty")

    rows = history[-args.last:]
    losses = [h["policy_loss"] for h in history]
    lo, hi = min(losses), max(losses)

    print(f"{'iter':>5} {'games':>10} {'p-loss':>8} {'v-loss':>8} {'top1':>6} "
          f"{'plies':>6} {'vs best':>8} {'':>3} {'policy loss':<18}")
    print("-" * 88)
    for h in rows:
        flag = "^" if h["promoted"] else " "
        print(
            f"{h['iteration']:>5} {h['games']:>10,} "
            f"{h['policy_loss']:>8.4f} {h['value_loss']:>8.4f} "
            f"{100 * h['policy_top1']:>5.1f}% {h['mean_plies']:>6.1f} "
            f"{100 * h['score_vs_best']:>7.1f}% {flag:>3} "
            f"{bar(h['policy_loss'], lo, hi):<18}"
        )

    promoted = sum(1 for h in history if h["promoted"])
    total_hours = sum(h["seconds"] for h in history) / 3600
    cum_elo = sum(h["elo_gain"] for h in history if h["promoted"])
    print("-" * 88)
    print(f"iterations   : {len(history)}  ({promoted} promoted, "
          f"{100 * promoted / len(history):.0f}%)")
    print(f"positions    : {history[-1]['games']:,}")
    print(f"elapsed      : {total_hours:.1f} h")
    print(f"cumulative Elo gain over the bootstrap: ~{cum_elo:+.0f}")
    if promoted == 0:
        print("\n  no promotions yet -- if this persists, the candidate is not")
        print("  beating the incumbent: try more steps/iter or a lower LR.")
    elif promoted < len(history) * 0.25:
        print("\n  low promotion rate -- training may be outpacing the data;")
        print("  consider more games per iteration or fewer training steps.")


if __name__ == "__main__":
    main()
