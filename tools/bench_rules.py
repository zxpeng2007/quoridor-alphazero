"""Benchmark the Numba rules core. Throughput here sets the MCTS budget."""

from __future__ import annotations

import time

import numpy as np

from quoridor import fastrules as fr


def collect_positions(n_games: int = 300, sample_rate: float = 0.2) -> list[np.ndarray]:
    """Realistic mid-game positions (walls on the board cost more to validate)."""
    rng = np.random.default_rng(0)
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    out: list[np.ndarray] = []
    for _ in range(n_games):
        st = fr.initial_state()
        for ply in range(250):
            if fr.winner(st) >= 0:
                break
            fr.legal_mask(st, mask, scratch)
            acts = np.nonzero(mask)[0]
            if len(acts) == 0:
                break
            if ply > 6 and rng.random() < sample_rate:
                out.append(st.copy())
            fr.apply_action(st, int(rng.choice(acts)), scratch)
    return out


def timeit(fn, iters: int) -> float:
    t0 = time.perf_counter()
    fn(iters)
    return time.perf_counter() - t0


def main() -> None:
    print("compiling (JIT warmup)...", flush=True)
    t0 = time.perf_counter()
    fr.warmup()
    print(f"  warmup: {time.perf_counter() - t0:.1f}s\n")

    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)

    positions = collect_positions()
    walls = [int(p[fr.WH_OFF:fr.WH_OFF + 128].sum()) for p in positions]
    print(f"sampled {len(positions)} mid-game positions, "
          f"mean walls on board = {np.mean(walls):.1f}\n")

    stack = np.stack(positions)
    n = len(stack)

    # --- legal_mask over realistic positions
    iters = 200_000
    t0 = time.perf_counter()
    for i in range(iters):
        fr.legal_mask(stack[i % n], mask, scratch)
    dt = time.perf_counter() - t0
    print(f"legal_mask      : {iters / dt / 1e6:.2f} M calls/s  ({dt / iters * 1e9:.0f} ns/call)")

    # --- opening position (empty board; cheapest case)
    st0 = fr.initial_state()
    t0 = time.perf_counter()
    for _ in range(iters):
        fr.legal_mask(st0, mask, scratch)
    dt = time.perf_counter() - t0
    print(f"legal_mask (open): {iters / dt / 1e6:.2f} M calls/s  ({dt / iters * 1e9:.0f} ns/call)")

    # --- shortest path
    t0 = time.perf_counter()
    for i in range(iters):
        fr.shortest_path_len(stack[i % n], 0, scratch)
    dt = time.perf_counter() - t0
    print(f"shortest_path   : {iters / dt / 1e6:.2f} M calls/s  ({dt / iters * 1e9:.0f} ns/call)")

    # --- full random playouts
    games = 20_000
    t0 = time.perf_counter()
    plies = 0
    for g in range(games):
        st = fr.initial_state()
        fr.random_playout(st, scratch, 400, g * 7919 + 13)
        plies += 1
    dt = time.perf_counter() - t0
    print(f"random_playout  : {games / dt:.0f} games/s")

    print("\nMCTS budget estimate:")
    t0 = time.perf_counter()
    for i in range(iters):
        fr.legal_mask(stack[i % n], mask, scratch)
        fr.apply_action(stack[i % n].copy(), 129, scratch)
    dt = time.perf_counter() - t0
    sims_per_s = iters / dt
    print(f"  legal_mask + apply + copy: {sims_per_s / 1e6:.2f} M/s per core")
    print(f"  across 16 cores          : {sims_per_s * 16 / 1e6:.1f} M/s")
    print("  (GPU inference, not rules, will be the self-play bottleneck)")


if __name__ == "__main__":
    main()
