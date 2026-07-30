"""Local training GUI server.

Serves the single-page board UI and a small JSON API around one
:class:`quoridor.webgui.GameSession`. Standard library only -- no framework.

    python tools/gui.py                        # play checkpoints/best.pt
    python tools/gui.py --checkpoint gen.pt --port 8630 --device cpu

Binds to 127.0.0.1 only: this is a personal trainer, not a network service.
The "reload engine" button re-reads the checkpoint from disk, so a training run
writing new promotions to best.pt can be picked up mid-session.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from quoridor import fastrules as fr
from quoridor.webgui import GameSession

STATIC_DIR = Path(__file__).resolve().parent / "gui_static"

_session: GameSession | None = None
_reload = None  # () -> dict, swaps the evaluator and returns checkpoint meta


def _load_evaluator(checkpoint: str, device: str):
    """Load the network; fall back to a uniform evaluator if nothing exists."""
    import torch

    from quoridor.net import NetEvaluator, UniformEvaluator, load_checkpoint

    path = Path(checkpoint)
    if not path.exists():
        fallback = path.parent / "bootstrap.pt"
        if fallback.exists():
            path = fallback
        else:
            print(f"warning: no checkpoint at {checkpoint}; engine plays without a net")
            return UniformEvaluator(), {"checkpoint": "none"}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    net, meta = load_checkpoint(path, device)
    meta = {**meta, "checkpoint": path.name, "device": device}
    # The GUI searches one game, so every simulation is a batch-1 network call;
    # capturing that shape as a CUDA graph makes the engine ~4x faster per move
    # with bit-identical outputs.
    return NetEvaluator(net, device=device, graph_batches=(1,)), meta


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._html(STATIC_DIR / "index.html")
        elif self.path == "/api/state":
            self._json(_session.snapshot())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        try:
            if self.path == "/api/new":
                _session.new_game(
                    human_player=int(body.get("human_player", 0)),
                    level=str(body.get("level", "medium")),
                )
            elif self.path == "/api/move":
                _session.human_move(int(body["action"]))
            elif self.path == "/api/reply":
                _session.engine_reply()  # slow call; idempotent if not engine's turn
            elif self.path == "/api/takeback":
                _session.takeback()
            elif self.path == "/api/resign":
                _session.resign()
            elif self.path == "/api/config":
                _session.set_config(
                    level=body.get("level"), coach=body.get("coach")
                )
            elif self.path == "/api/hint":
                self._json({"hint": _session.hint(), **_session.snapshot()})
                return
            elif self.path == "/api/reload":
                meta = _reload()
                self._json({"reloaded": meta, **_session.snapshot()})
                return
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json(_session.snapshot())
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc), **_session.snapshot()}, 400)
        except Exception as exc:  # keep the session alive on unexpected errors
            self._json({"error": f"internal: {exc}"}, 500)


def main() -> None:
    global _session, _reload

    ap = argparse.ArgumentParser(description="Barricade training GUI")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--port", type=int, default=8630)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    print("compiling rules engine (first run takes ~1 min, then cached)...")
    fr.warmup()
    print(f"loading {args.checkpoint} ...")
    evaluator, meta = _load_evaluator(args.checkpoint, args.device)

    _session = GameSession(evaluator, seed=args.seed)
    _session.meta = meta

    def reload_evaluator() -> dict:
        ev, m = _load_evaluator(args.checkpoint, args.device)
        _session.reload_evaluator(ev, m)
        return m

    _reload = reload_evaluator

    # Warm the search path so the first real move is instant.
    _session.hint_sims, keep = 16, _session.hint_sims
    _session.hint()
    _session.hint_sims = keep

    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  Barricade Trainer ready -> {url}")
    print(f"  engine: {meta.get('checkpoint')} on {meta.get('device', 'n/a')}"
          + (f" (iteration {meta['iteration']}, {meta['elo']:+.0f} Elo)"
             if "elo" in meta else ""))
    print("  Ctrl+C to stop\n")
    if not args.no_open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
