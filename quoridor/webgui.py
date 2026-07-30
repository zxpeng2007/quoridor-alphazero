"""Game-session logic behind the local training GUI.

The HTTP layer in ``tools/gui.py`` is a thin shell around :class:`GameSession`;
everything stateful lives here so it can be unit-tested without a server.

Design notes
------------
* One session, one game, one lock. The GUI is a single-user local app; every
  public method takes the lock because the Numba scratch buffers and the MCTS
  arena are not safe under concurrent calls.
* Engine replies are synchronous: ``human_move`` applies the human's action and,
  if the game is still live, immediately computes and applies the engine's
  response. The frontend shows a "thinking" state for the duration of the POST.
* All evals exposed to the UI are from the *human's* perspective, in [-1, 1].
  Internally MCTS values are side-to-move; conversions happen here, in one
  place, so the frontend never sees a sign flip.

Coach mode
----------
When enabled, every human turn gets a light evaluation search *before* the move
is played. That gives three things at once: a live eval bar, the engine's
suggested move for later review, and blunder detection -- the played move is
tagged when the post-move eval drops far below the pre-move eval.
"""

from __future__ import annotations

import threading

import numpy as np

from quoridor import fastrules as fr
from quoridor.mcts import BatchedMCTS

# Engine strength presets. Temperature > 0 samples from the visit distribution,
# which both weakens and varies play -- right for practice opponents.
LEVELS: dict[str, dict[str, float]] = {
    "beginner": {"sims": 24, "temperature": 1.0},
    "easy": {"sims": 96, "temperature": 0.6},
    "medium": {"sims": 320, "temperature": 0.0},
    "hard": {"sims": 800, "temperature": 0.0},
    "max": {"sims": 2000, "temperature": 0.0},
}
MAX_SIMS = max(int(cfg["sims"]) for cfg in LEVELS.values())

EVAL_SIMS = 192   # coach-mode evaluation search at each human turn
HINT_SIMS = 800   # explicit hint requests get a deeper look

# Opening variety: sample lightly for the first few plies so the engine does not
# play the identical opening every game (except at full strength).
OPENING_TEMP = 0.3
OPENING_PLIES = 6

# Blunder thresholds on the eval drop (human perspective, [-1, 1] scale).
BLUNDER_DROP = 0.5
INACCURACY_DROP = 0.22
# Below this pre-move eval the game is already lost; stop tagging noise.
LOST_ANYWAY = -0.8

FILES = "abcdefghi"


def cell_name(cell: int) -> str:
    r, c = divmod(int(cell), 9)
    return f"{FILES[c]}{r + 1}"


def action_text(action: int, st_after: np.ndarray | None = None,
                mover: int | None = None) -> str:
    """Human-readable move text. Pawn moves name their destination square."""
    action = int(action)
    if action >= fr.MOVE_BASE:
        if st_after is None or mover is None:
            return "move"
        dest = st_after[fr.IDX_P0 if mover == 0 else fr.IDX_P1]
        return f"→ {cell_name(int(dest))}"
    orient, slot = divmod(action, fr.NUM_WALL_SLOTS)
    wr, wc = divmod(slot, fr.W)
    glyph = "═" if orient == 0 else "║"  # ═ / ║
    return f"{glyph} {FILES[wc]}{wr + 1}"


class GameSession:
    """A single human-vs-engine game with coaching, undo, and hints."""

    def __init__(self, evaluator, seed: int = 0):
        self.lock = threading.RLock()
        self.mcts = BatchedMCTS(
            evaluator, n_games=1, max_nodes=MAX_SIMS + 128, seed=seed
        )
        self.scratch = fr.make_scratch()
        self.rng = np.random.default_rng(seed)
        self.meta: dict = {}
        self.coach = True
        self.eval_sims = EVAL_SIMS
        self.hint_sims = HINT_SIMS
        self.new_game(human_player=0, level="medium")

    # ------------------------------------------------------------- lifecycle

    def new_game(self, human_player: int = 0, level: str = "medium") -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        if human_player not in (0, 1):
            raise ValueError("human_player must be 0 or 1")
        with self.lock:
            self.state = fr.initial_state()
            self.history = [self.state.copy()]
            self.log: list[dict] = []
            self.result: int | None = None
            self.resigned = False
            self.human = int(human_player)
            self.level = level
            self.current_eval: float | None = None
            if int(self.state[fr.IDX_TURN]) != self.human:
                self._engine_move()

    def set_config(self, level: str | None = None, coach: bool | None = None) -> None:
        with self.lock:
            if level is not None:
                if level not in LEVELS:
                    raise ValueError(f"unknown level {level!r}")
                self.level = level
            if coach is not None:
                self.coach = bool(coach)

    def reload_evaluator(self, evaluator, meta: dict | None = None) -> None:
        """Swap in a newer network (e.g. after a training promotion) mid-session."""
        with self.lock:
            self.mcts.evaluator = evaluator
            self.meta = meta or {}

    # ----------------------------------------------------------------- moves

    def human_move(self, action: int) -> None:
        with self.lock:
            action = int(action)
            if self.result is not None:
                raise ValueError("the game is over")
            if int(self.state[fr.IDX_TURN]) != self.human:
                raise ValueError("not your turn")
            mask, _ = self._legal()
            if not (0 <= action < fr.NUM_ACTIONS) or not mask[action]:
                raise ValueError(f"illegal action {action}")

            pre_eval = None
            suggested = None
            if self.coach:
                # Evaluate before the move: powers the eval bar, the blunder
                # tag, and the "best was ..." note in the move list.
                a, v, _ = self._search(self.eval_sims)
                pre_eval, suggested = v, a  # side to move == human, no flip

            fr.apply_action(self.state, action, self.scratch)
            self.history.append(self.state.copy())
            entry = {
                "ply": len(self.log),
                "mover": self.human,
                "actor": "you",
                "action": action,
                "text": action_text(action, self.state, self.human),
                "eval": None,
                "tag": None,
                "best": None,
            }
            self.log.append(entry)

            w = fr.winner(self.state)
            if w >= 0:
                self.result = int(w)
                entry["eval"] = 1.0 if w == self.human else -1.0
                self.current_eval = entry["eval"]
                return

            post_eval = self._engine_move()  # human-perspective, after reply search
            entry["eval"] = post_eval
            if pre_eval is not None and post_eval is not None:
                drop = post_eval - pre_eval
                if pre_eval > LOST_ANYWAY and drop <= -BLUNDER_DROP:
                    entry["tag"] = "blunder"
                elif pre_eval > LOST_ANYWAY and drop <= -INACCURACY_DROP:
                    entry["tag"] = "inaccuracy"
                if entry["tag"] and suggested is not None and suggested != action:
                    # Replay the suggestion from the position it was searched in:
                    # history now ends [pre_move, after_human, after_engine], so
                    # the pre-move state (human to move) is at -3. Using -2 would
                    # apply the move as the ENGINE and then label the human's own
                    # blunder square as "best".
                    best_after = self.history[-3].copy()
                    fr.apply_action(best_after, int(suggested), self.scratch)
                    entry["best"] = action_text(int(suggested), best_after, self.human)

    def _engine_move(self) -> float | None:
        """Search, pick, and apply the engine's reply. Returns the human-persp eval."""
        cfg = LEVELS[self.level]
        ply = len(self.log)
        temp = float(cfg["temperature"])
        if ply < OPENING_PLIES and self.level != "max":
            temp = max(temp, OPENING_TEMP)

        action, value, visits = self._search(int(cfg["sims"]), temperature=temp)
        # `value` is from the engine's (side to move) perspective here.
        human_eval = -value

        fr.apply_action(self.state, action, self.scratch)
        self.history.append(self.state.copy())
        self.log.append({
            "ply": len(self.log),
            "mover": 1 - self.human,
            "actor": "engine",
            "action": action,
            "text": action_text(action, self.state, 1 - self.human),
            "eval": human_eval,
            "tag": None,
            "best": None,
        })
        self.current_eval = human_eval

        w = fr.winner(self.state)
        if w >= 0:
            self.result = int(w)
            self.current_eval = 1.0 if w == self.human else -1.0
        return human_eval

    def takeback(self) -> None:
        """Undo back to the most recent earlier position where it is the human's turn."""
        with self.lock:
            if not any(e["actor"] == "you" for e in self.log):
                raise ValueError("nothing to take back")
            while len(self.history) > 1:
                self.history.pop()
                self.log.pop()
                if int(self.history[-1][fr.IDX_TURN]) == self.human:
                    break
            self.state = self.history[-1].copy()
            self.result = None
            self.resigned = False
            self.current_eval = None

    def resign(self) -> None:
        with self.lock:
            if self.result is not None:
                raise ValueError("the game is over")
            self.result = 1 - self.human
            self.resigned = True

    # ----------------------------------------------------------------- coach

    def hint(self) -> dict:
        with self.lock:
            if self.result is not None:
                raise ValueError("the game is over")
            if int(self.state[fr.IDX_TURN]) != self.human:
                raise ValueError("not your turn")
            _, value, visits = self._search(self.hint_sims)
            total = float(visits.sum())
            order = np.argsort(-visits)[:5]
            _, dests = self._legal()
            top = []
            for a in order:
                if visits[a] <= 0:
                    continue
                a = int(a)
                dest = int(dests[a - fr.MOVE_BASE]) if a >= fr.MOVE_BASE else None
                top.append({
                    "action": a,
                    "dest": dest,
                    "share": float(visits[a] / total),
                    "text": action_text(a) if a < fr.MOVE_BASE else (
                        f"→ {cell_name(dest)}" if dest is not None else "move"
                    ),
                })
            return {"eval": float(value), "top": top}

    # -------------------------------------------------------------- internal

    def _legal(self) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(fr.NUM_ACTIONS, dtype=np.uint8)
        dests = np.zeros(12, dtype=np.int32)
        fr.legal_mask_dests(self.state, mask, self.scratch, dests)
        return mask, dests

    def _search(self, sims: int, temperature: float = 0.0):
        """Run a search from the current position; state is not mutated.

        Returns ``(chosen_action, root_value, visits)`` with the value from the
        side-to-move's perspective.
        """
        visits = self.mcts.search(self.state.reshape(1, -1).copy(), int(sims))[0]
        value = float(self.mcts.root_value()[0])
        if temperature > 1e-3 and visits.sum() > 0:
            p = np.power(visits, 1.0 / temperature)
            p = p / p.sum()
            action = int(self.rng.choice(len(p), p=p))
        else:
            action = int(visits.argmax())
        return action, value, visits

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> dict:
        with self.lock:
            st = self.state
            live_human_turn = (
                self.result is None and int(st[fr.IDX_TURN]) == self.human
            )
            legal_moves: list[dict] = []
            legal_walls: list[int] = []
            if live_human_turn:
                mask, dests = self._legal()
                for i in range(12):
                    if mask[fr.MOVE_BASE + i]:
                        legal_moves.append({
                            "action": fr.MOVE_BASE + i,
                            "dest": int(dests[i]),
                        })
                legal_walls = [int(a) for a in np.nonzero(mask[:fr.MOVE_BASE])[0]]

            return {
                "pawns": [int(st[fr.IDX_P0]), int(st[fr.IDX_P1])],
                "walls_h": [int(x) for x in st[fr.WH_OFF:fr.WH_OFF + 64]],
                "walls_v": [int(x) for x in st[fr.WV_OFF:fr.WV_OFF + 64]],
                "walls_left": [int(st[fr.IDX_WL0]), int(st[fr.IDX_WL1])],
                "turn": int(st[fr.IDX_TURN]),
                "ply": len(self.log),
                "human_player": self.human,
                "level": self.level,
                "coach": self.coach,
                "result": self.result,
                "resigned": self.resigned,
                "eval": self.current_eval,
                "legal_moves": legal_moves,
                "legal_walls": legal_walls,
                "log": list(self.log),
                "meta": dict(self.meta),
                "levels": list(LEVELS),
            }
