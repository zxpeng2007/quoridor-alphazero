"""Diagnose whether MCTS is actually searching or just echoing the prior.

If the root visit distribution collapses onto the network's top move everywhere,
the policy training target equals the current policy and self-play learns
nothing. This measures that directly: the entropy and top-move share of self-play
policy targets, plus how often search disagrees with the raw network prior.

A healthy run wants meaningful disagreement in mid-game positions -- the opening
is a forced race and near-deterministic play there is expected, not a problem.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from quoridor import fastrules as fr
from quoridor.encoding import NUM_PLANES, encode
from quoridor.net import NetEvaluator, load_checkpoint
from quoridor.selfplay import SelfPlayConfig, SelfPlayEngine


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=128)
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--fpu", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    fr.warmup()

    net, _ = load_checkpoint(args.checkpoint, device)
    ev = NetEvaluator(net, device=device)

    cfg = SelfPlayConfig(
        n_parallel=args.parallel, sims=args.sims, c_puct=args.c_puct, seed=0
    )
    engine = SelfPlayEngine(ev, cfg)
    engine.mcts.fpu_reduction = args.fpu
    batch = engine.generate(args.games, progress=True)
    print(f"\n{batch.stats}\n")

    pol = batch.policies.astype(np.float32)
    ent = entropy(pol)
    top = pol.max(axis=1)
    support = (pol > 0.01).sum(axis=1)

    print(f"{'':<22}{'mean':>8}{'p10':>8}{'p50':>8}{'p90':>8}")
    print("-" * 54)
    for label, arr in (
        ("policy entropy (nats)", ent),
        ("top-move share", top),
        ("moves above 1%", support.astype(np.float32)),
    ):
        print(f"{label:<22}{arr.mean():>8.3f}{np.percentile(arr, 10):>8.3f}"
              f"{np.percentile(arr, 50):>8.3f}{np.percentile(arr, 90):>8.3f}")

    collapsed = float((top > 0.98).mean())
    print(f"\nnear-deterministic targets (top move > 98%): {100 * collapsed:.1f}%")

    # How often does search disagree with the raw network prior?
    n = min(4000, len(batch.states))
    idx = np.random.default_rng(0).choice(len(batch.states), n, replace=False)
    planes = np.stack([encode(batch.states[i]) for i in idx]).astype(np.float32)
    legal = np.zeros((n, fr.NUM_ACTIONS), dtype=np.uint8)
    scratch = fr.make_scratch()
    for k, i in enumerate(idx):
        fr.legal_mask(batch.states[i], legal[k], scratch)
    prior, _ = ev.evaluate(planes, legal)
    # Both are in canonical space for the prior; compare argmax rank only.
    disagree = float((prior.argmax(1) != pol[idx].argmax(1)).mean())
    print(f"search disagrees with the raw prior on: {100 * disagree:.1f}% of positions")

    if collapsed > 0.9:
        print("\n  WARNING: search is echoing the prior almost everywhere.")
        print("  Self-play will not improve the policy in this state. Consider a")
        print("  higher c_puct, lower fpu, or more Dirichlet noise at the root.")
    elif collapsed > 0.6:
        print("\n  Search is concentrated but not fully collapsed; workable, though")
        print("  a higher c_puct would likely give a richer training signal.")
    else:
        print("\n  Healthy: targets carry entropy for search to learn from.")


if __name__ == "__main__":
    main()
