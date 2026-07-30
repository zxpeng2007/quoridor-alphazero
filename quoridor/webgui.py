"""Game-session logic behind the local training GUI.

The HTTP layer in ``tools/gui.py`` is a thin shell around :class:`GameSession`;
everything stateful lives here so it can be unit-tested without a server.

Design notes
------------
* One session, one game, one lock. The GUI is a single-user local app; every
  public method takes the lock because the Numba scratch buffers and the MCTS
  arena are not safe under concurrent calls.
* Moves are a **two-step flow** so the human's move renders immediately:
  ``human_move`` validates and applies the human's action and returns at once;
  the client then calls ``engine_reply`` (a separate, slow request) for the
  engine's response. ``engine_reply`` is idempotent -- if it is not actually the
  engine's turn (double request, page refresh mid-reply) it is a no-op.
* All evals exposed to the UI are from the *human's* perspective, in [-1, 1].
  Internally MCTS values are side-to-move; conversions happen here, in one
  place, so the frontend never sees a sign flip.

Coach mode
----------
The coach evaluation for the human's current position is **precomputed** the
moment their turn starts (end of the engine's reply, after an undo, on new
game), while the human is still thinking. ``human_move`` then reads the cached
result instead of searching, which keeps the human's own move instant. The
cache powers the live eval bar, the blunder tags, and the "best was ..." note;
a hint request refreshes it with a deeper search.
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

EVAL_SIMS = 192   # coach-mode evaluation search, precomputed per human turn
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
    glyph = "═" if orient == 0 else "║"
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
        """Reset. Does NOT play the engine's opening move when the engine is
        first -- the client requests that via :meth:`engine_reply`, so the
        initial board renders before the engine starts thinking."""
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
            # (pre-move eval, suggested action) cached for the human's turn.
            self._turn_eval: tuple[float, int] | None = None
            # Blunder-tagging context carried from human_move to engine_reply.
            self._pending: dict | None = None
            self._prepare_turn()

    def set_config(self, level: str | None = None, coach: bool | None = None) -> None:
        with self.lock:
            if level is not None:
                if level not in LEVELS:
                    raise ValueError(f"unknown level {level!r}")
                self.level = level
            if coach is not None:
                coach = bool(coach)
                turned_on = coach and not self.coach
                self.coach = coach
                self._turn_eval = None
                if turned_on:
                    self._prepare_turn()

    def reload_evaluator(self, evaluator, meta: dict | None = None) -> None:
        """Swap in a newer network (e.g. after a training promotion) mid-session."""
        with self.lock:
            self.mcts.evaluator = evaluator
            self.meta = meta or {}

    # ----------------------------------------------------------------- moves

    def human_move(self, action: int) -> None:
        """Validate and apply the human's move. Fast: no search happens here --
        the coach eval was precomputed when the turn started, and the engine's
        reply is a separate :meth:`engine_reply` call."""
        with self.lock:
            action = int(action)
            if self.result is not None:
                raise ValueError("the game is over")
            if int(self.state[fr.IDX_TURN]) != self.human:
                raise ValueError("not your turn")
            mask, _ = self._legal()
            if not (0 <= action < fr.NUM_ACTIONS) or not mask[action]:
                raise ValueError(f"illegal action {action}")

            pre_eval: float | None = None
            suggested: int | None = None
            if self.coach:
                if self._turn_eval is None:  # rare fallback (e.g. toggled mid-turn)
                    self._prepare_turn()
                if self._turn_eval is not None:
                    pre_eval, suggested = self._turn_eval

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
            self._turn_eval = None

            w = fr.winner(self.state)
            if w >= 0:
                self.result = int(w)
                entry["eval"] = 1.0 if w == self.human else -1.0
                self.current_eval = entry["eval"]
                self._pending = None
                return

            self._pending = {
                "entry": entry, "pre_eval": pre_eval, "suggested": suggested,
            }

    def engine_reply(self) -> None:
        """Compute and apply the engine's move (the slow call).

        Idempotent: a duplicate request -- double click, page refresh while a
        reply was already in flight -- finds it is not the engine's turn and
        returns without error.
        """
        with self.lock:
            if self.result is not None:
                return
            if int(self.state[fr.IDX_TURN]) == self.human:
                return

            cfg = LEVELS[self.level]
            temp = float(cfg["temperature"])
            if len(self.log) < OPENING_PLIES and self.level != "max":
                temp = max(temp, OPENING_TEMP)
            action, value, _ = self._search(int(cfg["sims"]), temperature=temp)
            # `value` is from the engine's (side to move) perspective.
            human_eval = -value

            # Settle the pending human entry: eval, blunder tag, "best was".
            if self._pending is not None:
                entry = self._pending["entry"]
                pre_eval = self._pending["pre_eval"]
                suggested = self._pending["suggested"]
                entry["eval"] = human_eval
                if pre_eval is not None:
                    drop = human_eval - pre_eval
                    if pre_eval > LOST_ANYWAY and drop <= -BLUNDER_DROP:
                        entry["tag"] = "blunder"
                    elif pre_eval > LOST_ANYWAY and drop <= -INACCURACY_DROP:
                        entry["tag"] = "inaccuracy"
                if entry["tag"] and suggested is not None and suggested != entry["action"]:
                    # Replay the suggestion from the position it was searched in.
                    # The engine's move is not applied yet, so history ends
                    # [..., pre_move, after_human] and the pre-move state
                    # (human to move) is at -2.
                    best_after = self.history[-2].copy()
                    fr.apply_action(best_after, int(suggested), self.scratch)
                    entry["best"] = action_text(int(suggested), best_after, self.human)
                self._pending = None

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
                return
            # Precompute the coach eval for the human's next turn -- but off the
            # reply's critical path, so the engine's move renders as soon as its
            # own search finishes. If the human moves before this lands,
            # human_move's fallback computes it synchronously, same as before.
            threading.Thread(
                target=self._prepare_turn_async, args=(len(self.log),), daemon=True
            ).start()

    def _prepare_turn_async(self, expected_ply: int) -> None:
        with self.lock:
            # The position may have changed while this thread waited on the
            # lock (fast human move, undo, new game) -- only fill the cache if
            # it is still the same turn it was scheduled for.
            if len(self.log) == expected_ply and self._turn_eval is None:
                self._prepare_turn()

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
            self._pending = None
            self._turn_eval = None
            self._prepare_turn()

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
            action, value, visits = self._search(self.hint_sims)
            # A hint is a deeper search than the precomputed eval; adopt it as
            # the turn's coach context so tagging uses the better numbers.
            self._turn_eval = (value, action)
            self.current_eval = value
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

    def _prepare_turn(self) -> None:
        """Cache (eval, suggestion) for the human's current turn, if coaching."""
        if not self.coach or self.result is not None:
            return
        if int(self.state[fr.IDX_TURN]) != self.human:
            return
        action, value, _ = self._search(self.eval_sims)
        self._turn_eval = (value, action)
        self.current_eval = value

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
                # True when the client should follow up with /api/reply.
                "needs_reply": (
                    self.result is None and int(st[fr.IDX_TURN]) != self.human
                ),
                "legal_moves": legal_moves,
                "legal_walls": legal_walls,
                "log": list(self.log),
                "meta": dict(self.meta),
                "levels": list(LEVELS),
            }
