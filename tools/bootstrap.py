"""Supervised bootstrap: pretrain the network on heuristic-agent games.

Why this stage exists
---------------------
Quoridor has up to 140 legal actions. Starting AlphaZero from a randomly
initialised policy means MCTS begins with an essentially flat prior, and at any
affordable simulation count the root visit counts come out near-uniform -- so the
policy target carries almost no signal and training barely moves. (Turning on FPU
reduction to concentrate the search makes it worse, not better: with a flat prior
the penalty swamps every exploration term and search tunnels into whichever action
it happens to try first.)

Pretraining on alpha-beta games breaks that deadlock cheaply. The heuristic is far
from strong, so this is a *starting point*, not a teacher -- self-play is what
actually produces strength, and it will train through and past this policy. The
point is only to hand MCTS a prior peaked enough to be worth searching with.

Positions are collected DAgger-style: the heuristic's preferred move is always
recorded as the target, but a fraction of moves actually played are random, which
spreads the data over positions the heuristic would never reach on its own.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from numba import njit, prange

from quoridor import fastrules as fr
from quoridor.agents import HeuristicAgent
from quoridor.net import NetConfig, QuoridorNet, save_checkpoint
from quoridor.replay import ReplayBuffer
from quoridor.train import Trainer, TrainConfig


def _worker(args):
    """Generate games in one process; returns (states, actions, values)."""
    n_games, depth, epsilon, seed, max_plies = args
    rng = np.random.default_rng(seed)
    scratch = fr.make_scratch()
    mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
    agents = [HeuristicAgent(depth=depth, seed=seed), HeuristicAgent(depth=depth, seed=seed + 7919)]

    out_states: list[np.ndarray] = []
    out_actions: list[int] = []
    out_values: list[np.ndarray] = []

    for _ in range(n_games):
        st = fr.initial_state()
        states: list[np.ndarray] = []
        actions: list[int] = []
        players: list[int] = []
        for _ply in range(max_plies):
            if fr.winner(st) >= 0:
                break
            turn = int(st[fr.IDX_TURN])
            best = int(agents[turn].select_action(st))
            states.append(st.copy())
            actions.append(best)
            players.append(turn)

            played = best
            if rng.random() < epsilon:  # exploration: widen the state distribution
                fr.legal_mask(st, mask, scratch)
                played = int(rng.choice(np.nonzero(mask)[0]))
            fr.apply_action(st, played, scratch)

        w = fr.winner(st)
        if w < 0:
            continue  # unfinished games have no value target
        out_states.extend(states)
        out_actions.extend(actions)
        out_values.append(
            np.array([1.0 if p == w else -1.0 for p in players], dtype=np.float32)
        )

    if not out_states:
        return (
            np.zeros((0, fr.STATE_SIZE), np.uint8),
            np.zeros(0, np.int16),
            np.zeros(0, np.float32),
        )
    return (
        np.stack(out_states),
        np.asarray(out_actions, dtype=np.int16),
        np.concatenate(out_values),
    )


def generate(games: int, depth: int, epsilon: float, workers: int, max_plies: int):
    per = max(games // workers, 1)
    jobs = [(per, depth, epsilon, 1000 + i, max_plies) for i in range(workers)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(_worker, jobs))
    dt = time.perf_counter() - t0

    states = np.concatenate([c[0] for c in chunks])
    actions = np.concatenate([c[1] for c in chunks])
    values = np.concatenate([c[2] for c in chunks])
    print(
        f"generated {per * workers} games -> {len(states):,} positions "
        f"in {dt:.0f}s ({per * workers / dt:.0f} games/s)"
    )
    return states, actions, values


@njit(cache=True, parallel=True)
def _smooth_policies(states, actions, smoothing, out):
    """Spread ``smoothing`` mass uniformly over legal moves, rest on the choice."""
    B = states.shape[0]
    for i in prange(B):
        scratch = np.zeros(fr.SCRATCH_SIZE, dtype=np.int32)
        mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
        fr.legal_mask(states[i], mask, scratch)
        n = 0
        for a in range(fr.NUM_ACTIONS):
            n += mask[a]
        share = smoothing / n if n > 0 else 0.0
        for a in range(fr.NUM_ACTIONS):
            out[i, a] = share * mask[a]
        out[i, actions[i]] += 1.0 - smoothing


def fill_buffer(
    buf: ReplayBuffer, states, actions, values, smoothing: float = 0.15,
    chunk: int = 200_000,
):
    """Expand integer actions into policy targets, in chunks to cap peak memory.

    Targets are label-smoothed rather than one-hot. Training to one-hot drives the
    softmax to near-certainty, and an overconfident prior makes MCTS collapse onto
    its top move -- at which point search returns the policy it was given, the
    "improved" target equals the current policy, and self-play training has
    nothing to learn from. Keeping some mass on the other legal moves leaves
    search room to actually disagree with the prior.
    """
    for i in range(0, len(states), chunk):
        sl = slice(i, i + chunk)
        chunk_states = np.ascontiguousarray(states[sl])
        acts = actions[sl].astype(np.int64)
        pol = np.zeros((len(acts), fr.NUM_ACTIONS), dtype=np.float32)
        if smoothing > 0:
            _smooth_policies(chunk_states, acts, smoothing, pol)
        else:
            pol[np.arange(len(acts)), acts] = 1.0
        buf.add(chunk_states, pol.astype(np.float16), values[sl])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30_000)
    ap.add_argument("--depth", type=int, default=2, help="heuristic search depth")
    ap.add_argument("--epsilon", type=float, default=0.20, help="random-move rate")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--out", default="checkpoints/bootstrap.pt")
    ap.add_argument("--save-data", default="", help="optional .npz path for raw games")
    ap.add_argument("--load-data", default="", help="reuse raw games from a .npz")
    ap.add_argument("--label-smoothing", type=float, default=0.15)
    args = ap.parse_args()

    fr.warmup()
    if args.load_data:
        blob = np.load(args.load_data)
        states, actions, values = blob["states"], blob["actions"], blob["values"]
        print(f"loaded {len(states):,} positions from {args.load_data}")
    else:
        states, actions, values = generate(
            args.games, args.depth, args.epsilon, args.workers, args.max_plies
        )
        if args.save_data:
            Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
            # Save raw actions, not expanded policies, so the smoothing level can
            # be changed later without regenerating games.
            np.savez_compressed(
                args.save_data, states=states, actions=actions, values=values
            )
            print(f"saved raw games -> {args.save_data}")

    buf = ReplayBuffer(capacity=len(states) + 16)
    fill_buffer(buf, states, actions, values, smoothing=args.label_smoothing)
    print(f"replay buffer: {len(buf):,} positions "
          f"(label smoothing {args.label_smoothing})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    net = QuoridorNet(NetConfig(channels=args.channels, blocks=args.blocks))
    print(f"net: {args.blocks}x{args.channels}, {net.num_params() / 1e6:.2f}M params\n")

    trainer = Trainer(
        net,
        TrainConfig(batch_size=args.batch_size, lr=args.lr),
        device=device,
        total_steps=args.steps,
    )
    t0 = time.perf_counter()
    final = trainer.train_on_buffer(buf, args.steps, log_every=max(args.steps // 20, 1))
    dt = time.perf_counter() - t0

    print(f"\nsupervised training done in {dt / 60:.1f} min")
    print(f"final: {final}")
    save_checkpoint(
        args.out,
        net,
        meta={
            "stage": "bootstrap",
            "games": args.games,
            "positions": len(buf),
            "steps": args.steps,
            "heuristic_depth": args.depth,
            "policy_top1": final.policy_acc,
            "value_mae": final.value_mae,
        },
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
