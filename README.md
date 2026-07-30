# Quoridor AlphaZero

A self-play reinforcement learning engine for Quoridor — the game played at
[barricade.gg](https://barricade.gg/). AlphaZero-style: a policy/value ResNet
guided by PUCT Monte Carlo tree search, trained from self-play.

Built to run on a single machine (developed on a Ryzen 9 9950X3D + RTX 5080).

## Quick start

```bash
pip install -e .
```

```bash
python -m pytest tests/
```

Bootstrap a starting network from alpha-beta games (see *Why bootstrap* below),
then run self-play training:

```bash
python tools/bootstrap.py --games 60000 --depth 2 --steps 8000 --save-data data/heuristic_games.npz
```

```bash
python tools/train.py --init checkpoints/bootstrap.pt --iterations 100
```

Check on a run in progress:

```bash
python tools/status.py --dir checkpoints
```

Play against it, or analyse a position:

```bash
python tools/play.py checkpoints/best.pt play --first --sims 800
```

```bash
python tools/play.py checkpoints/best.pt analyse --moves "S,N,S,N"
```

## Training GUI

A local graphical trainer for practising against the engine:

```bash
python tools/gui.py
```

Opens `http://127.0.0.1:8630/` in your browser (standard library server, binds
localhost only). Features:

- click-to-move with legal-move dots; hover a groove to preview a wall
- five strength levels, from sampled-and-shallow up to 2000 sims/move
- **coach mode**: live eval bar, blunder/inaccuracy tags on your moves with the
  engine's suggestion, and a hint button (deeper search, shown as a ghost)
- undo, resign, board flip, engine-first games
- *reload engine* button re-reads the checkpoint mid-session, so a training run
  writing new promotions to `best.pt` can be picked up between games

The session logic lives in `quoridor/webgui.py` and is covered by
`tests/test_webgui.py`; `tools/gui.py` is a thin stdlib HTTP shell around it.

## Layout

| Module | Purpose |
| --- | --- |
| `quoridor/pyrules.py` | Readable reference rules — the correctness ground truth |
| `quoridor/fastrules.py` | Numba-jitted core: move generation, wall legality, BFS |
| `quoridor/encoding.py` | Network input planes, canonicalisation, symmetries |
| `quoridor/mcts.py` | Batched PUCT MCTS over many concurrent games |
| `quoridor/net.py` | Policy/value ResNet and its batched GPU evaluator |
| `quoridor/selfplay.py` | Self-play game generation |
| `quoridor/replay.py` | Replay buffer and training-batch encoder |
| `quoridor/train.py` | Loss and optimiser step |
| `quoridor/arena.py` | Match play, adjudication, and Elo estimation |
| `quoridor/agents.py` | Baseline agents (random, greedy, alpha-beta heuristic) |

## Design notes

Things that turned out to matter, recorded because most of them were not obvious
up front.

### Wall legality is the hot path, and the empty board is the worst case

A wall may not seal any player off from their goal, so every candidate placement
needs a connectivity check. Two optimisations take this from 93 µs to ~7 µs per
position:

1. **Path witness.** Compute one shortest path per player and mark its edges. A
   wall that cuts no marked edge cannot disconnect that player — the witnessed
   path survives — so only path-crossing candidates need a real search, and only
   for the player whose path they cross.
2. **Bitboard flood fill.** The remaining checks use an 81-bit occupancy set in
   two `uint64` words, expanding the whole frontier in a few shifts instead of
   walking a queue.

Counter-intuitively the *empty* board was originally the slowest case: with no
walls placed both pawns' shortest paths run straight down column 4, so ~16
candidate walls cross a path and each triggered two full BFS runs.

### The 140-action space is the central difficulty

Quoridor has up to 140 legal actions (128 wall placements + 12 pawn moves), far
wider than chess. This has consequences that took measurement to see:

- **Knowledge-free search is nearly blind.** With a uniform prior, PUCT must try
  every action once before revisiting any, so at realistic simulation counts the
  root visit distribution comes out almost uniform — and the visit distribution
  *is* the policy training target. Starting from random weights, there is almost
  no signal to learn from.
- **FPU reduction cuts both ways.** Making unvisited moves pessimistic
  concentrates search, which is what wide action spaces need — but under a flat
  prior every exploration term is ~1/140 and the penalty swamps them, collapsing
  search onto whichever action it happens to try first. It helps only once the
  prior is peaked.

Together these are why `tools/bootstrap.py` exists.

### Why bootstrap

Pretraining on alpha-beta self-play games hands MCTS a prior peaked enough to be
worth searching with, which breaks the deadlock above. The heuristic is weak
(~1400 Elo on the internal scale) and is a *starting point*, not a teacher —
self-play trains through and past it. Positions are collected DAgger-style: the
heuristic's preferred move is always the target, but a fraction of moves actually
played are random, spreading the data beyond states the heuristic reaches itself.

This is a departure from AlphaZero's tabula-rasa purity, made deliberately: it
trades a little initial bias for a large amount of compute.

**Targets are label-smoothed, and this is load-bearing.** The first bootstrap
used one-hot targets, and the resulting network was so confident that MCTS put
*100% of 400 simulations on a single move*. That is a silent way to kill an
AlphaZero run: if search always returns the prior, the "improved" policy target
equals the current policy and there is nothing left to learn from. Keeping ~15%
of the mass spread over the other legal moves leaves search enough room to
disagree with the prior, which is the entire mechanism by which the network
improves.

### Adjudicating stalled games

Quoridor has no repetition rule, so a losing player can refuse to advance and
shuffle indefinitely. Measured against the heuristic, an early network drew half
its games this way by hitting the ply cap. Capped games are therefore awarded to
whoever is closer to their goal, which both resolves them and rewards making
progress.

### Symmetry

The left-right mirror is the *only* symmetry Quoridor admits — rotations and
transposes do not preserve the rules, because each player's goal row is fixed.
It is used for training-data augmentation. The 180° rotation is used separately
for canonicalisation, so the network only ever sees the side-to-move view.

Both act on the action space as permutations. A wrong entry would not crash
anything — it would quietly train the network to play good moves on a mirrored
board — so the tests check that both symmetries commute with legality *and*
dynamics over real positions.

### Elo with shutouts

A 40-0 result is a complete-separation case: the maximum-likelihood Elo
difference is infinite. The first baseline ladder duly reported "+4022 Elo ±
1702434". Virtual draws are now mixed into every pairing, which keeps estimates
finite and honest about how little a shutout actually pins down.

## Measured performance

Rules engine, single core:

| Operation | Rate |
| --- | --- |
| `legal_mask`, mid-game | 1.15 M/s |
| `legal_mask`, empty board | 145 k/s |
| Random playouts | 4.3 k games/s |

Network and search (RTX 5080, 8×128 net):

| | Rate |
| --- | --- |
| Inference, batch 1024 | 122 k positions/s |
| Self-play, 1024 concurrent games | ~30 k games/hour @ 200 sims |

Baseline ladder (`tools/eval_baselines.py`, random anchored at 0):

| Agent | Elo |
| --- | --- |
| heuristic-d3 | 1412 |
| heuristic-d2 | 995 |
| greedy / heuristic-d1 | 527 |
| random | 0 |

These are *internal* ratings from self-play among these agents. They are not on
the same scale as barricade.gg's ladder and should not be read as comparable to
it.

## Scope

This engine is for offline play, analysis, and training. It is not wired to
barricade.gg and is not intended to be: running a bot against human opponents on
a live ranked ladder is against the terms of essentially every such site, and the
people on the other end did not consent to playing a program.
