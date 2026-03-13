"""
Web-based game player with multi-backend support.

Wraps the JAX, NodeJS, and C++ PuzzleScript backends behind a unified Flask-SocketIO
server so any of them can be played in a browser from a remote machine.  The UI has:
  - Rendered game frame (updated via WebSocket)
  - Backend toggle (JAX / NodeJS / CPP)
  - Live console that mirrors server stdout / stderr

Usage (same CLI as human_env.py):
    python human_env_server.py game=Microban level=0 jit=false backend=jax
    python human_env_server.py game=Microban backend=nodejs
    python human_env_server.py game=Microban backend=cpp

Then open  http://<server-ip>:<port>/  in your local browser.

Key bindings (same as human_env.py):
    WASD / arrow keys  – move
    x                  – action / interact
    r                  – restart level
    n                  – next level
    b                  – previous level
    z                  – undo
    ESC                – quit (closes server)
"""
from __future__ import annotations

import base64
import glob
import io
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import hydra
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from hydra.core.config_store import ConfigStore
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    jit: bool = True
    game: Optional[str] = None
    profile: bool = False
    debug: bool = False
    level: int = 0
    port: int = 5000
    host: str = "0.0.0.0"
    backend: str = "jax"   # "jax" | "nodejs" | "cpp"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

SCALING_FACTOR = 4
MAX_AGAIN = 50

# ---------------------------------------------------------------------------
# Action mappings  (key name → engine-specific action index)
# ---------------------------------------------------------------------------

_ACTION_MAPS: dict[str, dict[str, int]] = {
    # human_env.py: a=left=0, s=down=1, d=right=2, w=up=3, x=4
    "jax":    {"left": 0, "down": 1, "right": 2, "up": 3, "x": 4},
    # Same convention as CPP (empirically verified: 0=up,1=left,2=down,3=right,4=action).
    # Note: llm_agent_loop_nodejs.py ACTION_MEANINGS appears to be incorrect.
    "nodejs": {"up": 0, "left": 1, "down": 2, "right": 3, "x": 4},
    # CppPuzzleScriptEngine.process_input docstring: 0=up,1=left,2=down,3=right,4=action
    "cpp":    {"up": 0, "left": 1, "down": 2, "right": 3, "x": 4},
}

# ---------------------------------------------------------------------------
# Stdout / stderr capture → forwarded to all browser consoles
# ---------------------------------------------------------------------------

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


class _FDCapture:
    """Redirects OS-level fds 1 and 2 to pipes and tees to original fds + SocketIO.

    This captures *all* output including C-extension printf / JAX prints that
    bypass Python's sys.stdout wrapper.
    """

    def __init__(self):
        import select as _select
        self._select = _select

        # Preserve originals so we can still tee to the terminal.
        self._orig_out = os.dup(1)
        self._orig_err = os.dup(2)

        self._r_out, w_out = os.pipe()
        self._r_err, w_err = os.pipe()

        os.dup2(w_out, 1); os.close(w_out)
        os.dup2(w_err, 2); os.close(w_err)

        # Make Python's buffered IO line-buffered so output flows quickly.
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name="fd-capture")
        self._thread.start()

    def _reader(self):
        fds = [self._r_out, self._r_err]
        while not self._stop.is_set():
            try:
                r, _, _ = self._select.select(fds, [], fds, 0.1)
            except (ValueError, OSError):
                break
            for fd in r:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    continue
                if not data:
                    continue
                orig = self._orig_out if fd == self._r_out else self._orig_err
                try:
                    os.write(orig, data)
                except OSError:
                    pass
                text = data.decode("utf-8", errors="replace")
                try:
                    socketio.emit("log", {"text": text})
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        os.dup2(self._orig_out, 1)
        os.dup2(self._orig_err, 2)
        for fd in (self._orig_out, self._orig_err, self._r_out, self._r_err):
            try:
                os.close(fd)
            except OSError:
                pass


_fd_capture: Optional[_FDCapture] = None


def _install_log_capture():
    global _fd_capture
    _fd_capture = _FDCapture()


# ---------------------------------------------------------------------------
# Abstract player-backend interface
# ---------------------------------------------------------------------------

class PlayerBackend(ABC):
    """Unified interface for all three PuzzleScript backends."""

    @abstractmethod
    def load_game(self, game: str, level_i: int) -> None: ...

    @abstractmethod
    def render_frame(self) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB array."""
        ...

    @abstractmethod
    def step(self, direction: str) -> None:
        """direction ∈ {'left','right','up','down','x'}"""
        ...

    @abstractmethod
    def reset_level(self) -> None: ...

    @abstractmethod
    def load_level(self, level_i: int) -> None: ...

    @abstractmethod
    def backup(self) -> Any: ...

    @abstractmethod
    def restore(self, snapshot: Any) -> None: ...

    @property
    @abstractmethod
    def winning(self) -> bool: ...

    @property
    @abstractmethod
    def num_levels(self) -> int: ...

    @property
    @abstractmethod
    def title(self) -> str: ...

    # Populated by load_game
    game_text: str = ""


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

class JaxPlayerBackend(PlayerBackend):
    _ACTION_MAP = _ACTION_MAPS["jax"]

    def __init__(self, jit: bool = False, debug: bool = False,
                 profile: bool = False):
        self._jit     = jit
        self._debug   = debug
        self._profile = profile
        self._env     = None
        self._state   = None
        self._params  = None
        self._rng     = None
        self._title   = "PuzzleScript"

    def load_game(self, game: str, level_i: int) -> None:
        import jax
        from lark import Lark
        from puzzlescript_jax.globals import LARK_SYNTAX_PATH
        from puzzlescript_jax.env import PuzzleJaxEnv, PJParams
        from puzzlescript_jax.preprocessing import get_tree_from_txt, PJParseErrors

        with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
            grammar = f.read()
        parser = Lark(grammar, start="ps_game", maybe_placeholders=False)

        print(f'[JAX] Parsing game: "{game}"')
        tree, success, err_msg = get_tree_from_txt(
            parser, game, overwrite=True, test_env_init=False)
        if success != PJParseErrors.SUCCESS:
            raise RuntimeError(f"JAX parse error: {err_msg}")

        print(f'[JAX] Initialising environment for: "{game}"')
        self._env    = PuzzleJaxEnv(tree, jit=self._jit, debug=self._debug,
                                    print_score=True)
        self._title  = self._env.title
        self._rng    = jax.random.PRNGKey(0)
        self._params = PJParams(level=self._env.get_level(level_i))
        self._obs, self._state = self._env.reset(self._rng, self._params)
        self._level_i = level_i
        self.game_text = game

    def render_frame(self) -> np.ndarray:
        im = self._env.render(self._state, cv2=False)  # RGBA
        return np.array(im, dtype=np.uint8)[:, :, :3]  # drop alpha

    def step(self, direction: str) -> None:
        import jax
        import jax.numpy as jnp
        from puzzlescript_jax.env import multihot_to_desc
        action_i = self._ACTION_MAP[direction]
        print(f"\n========= STEP =========")
        print(f"[JAX] action: {direction} (idx {action_i})")
        self._rng, _ = jax.random.split(self._rng)
        act = jnp.array(action_i, dtype=jnp.int32)
        self._obs, self._state, reward, done, info = self._env.step(
            self._rng, self._state, act, self._params)
        print(multihot_to_desc(self._state.multihot_level, self._env.objs_to_idxs,
                               self._env.n_objs,
                               obj_idxs_to_force_idxs=self._env.obj_idxs_to_force_idxs))
        print(f"win={bool(self._state.win)}  restart={bool(self._state.restart)}"
              f"  reward={float(reward):.3f}")

    def reset_level(self) -> None:
        import jax
        self._rng, _ = jax.random.split(self._rng)
        self._obs, self._state = self._env.reset(self._rng, self._params)

    def load_level(self, level_i: int) -> None:
        import jax
        from puzzlescript_jax.env import PJParams
        self._level_i = level_i
        self._params  = self._params.replace(level=self._env.get_level(level_i))
        self._rng, _  = jax.random.split(self._rng)
        self._obs, self._state = self._env.reset(self._rng, self._params)

    def backup(self) -> Any:
        return self._state

    def restore(self, snapshot: Any) -> None:
        self._state = snapshot

    @property
    def winning(self) -> bool:
        return bool(self._state.win)

    @property
    def restarting(self) -> bool:
        return bool(self._state.restart)

    @property
    def num_levels(self) -> int:
        return len(self._env.levels)

    @property
    def title(self) -> str:
        return self._title


# ---------------------------------------------------------------------------
# NodeJS backend
# ---------------------------------------------------------------------------

class NodeJSPlayerBackend(PlayerBackend):
    _ACTION_MAP = _ACTION_MAPS["nodejs"]

    def __init__(self):
        self._backend   = None
        self._game_text = ""
        self._title     = "PuzzleScript"
        self._level_i   = 0
        self._n_levels  = 1

    def _ensure_backend(self):
        if self._backend is None:
            from backends.nodejs import NodeJSPuzzleScriptBackend
            print("[NodeJS] Loading JS engine…")
            self._backend = NodeJSPuzzleScriptBackend()

    def load_game(self, game: str, level_i: int) -> None:
        from puzzlescript_jax.preprocessing import SIMPLIFIED_GAMES_DIR

        self._ensure_backend()
        print(f'[NodeJS] Compiling game: "{game}"')

        simp_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}_simplified.txt")
        if os.path.isfile(simp_path):
            # Simplified file already exists (e.g. written by the JAX backend).
            # Read it directly to avoid the SIGALRM-based timeout in get_tree_from_txt,
            # which cannot be used from a non-main thread.
            with open(simp_path, "r", encoding="utf-8") as f:
                self._game_text = f.read()
            self._backend.engine.compile(["restart"], self._game_text)
        else:
            # First time seeing this game — parse from scratch.
            # NOTE: this path only works safely on the main thread.
            from lark import Lark
            from puzzlescript_jax.globals import LARK_SYNTAX_PATH
            with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
                grammar = f.read()
            parser = Lark(grammar, start="ps_game", maybe_placeholders=False)
            self._game_text = self._backend.compile_game(parser, game)

        self._n_levels  = self._backend.get_num_levels()
        self._level_i   = level_i
        self._title     = game
        self._backend.load_level(self._game_text, level_i)
        self.game_text = self._game_text

    def render_frame(self) -> np.ndarray:
        return self._backend.render_frame()  # already RGB (H, W, 3)

    def step(self, direction: str) -> None:
        action_i = self._ACTION_MAP[direction]
        print(f"\n========= STEP =========")
        print(f"[NodeJS] action: {direction} (idx {action_i})")
        self._backend.engine.processInput(action_i)
        again_steps = 0
        while self._backend.engine.getAgaining() and again_steps < MAX_AGAIN:
            self._backend.engine.processInput(-1)
            again_steps += 1
        winning   = bool(self._backend.engine.getWinning())
        againing  = bool(self._backend.engine.getAgaining())
        score = None
        try:
            score = float(self._backend.solver.getScore(self._backend.engine))
        except Exception:
            pass
        score_str = f"  score={score:.3f}" if score is not None else ""
        print(f"win={winning}  againing={againing}{score_str}")

    def reset_level(self) -> None:
        self._backend.load_level(self._game_text, self._level_i)

    def load_level(self, level_i: int) -> None:
        self._level_i = level_i
        self._backend.load_level(self._game_text, level_i)

    def backup(self) -> Any:
        return self._backend.engine.backupLevel()

    def restore(self, snapshot: Any) -> None:
        self._backend.engine.restoreLevel(snapshot)

    @property
    def winning(self) -> bool:
        return bool(self._backend.engine.getWinning())

    @property
    def restarting(self) -> bool:
        return bool(self._backend.engine.getRestarting())

    @property
    def num_levels(self) -> int:
        return self._n_levels

    @property
    def title(self) -> str:
        return self._title


# ---------------------------------------------------------------------------
# CPP backend
# ---------------------------------------------------------------------------

class CppPlayerBackend(PlayerBackend):
    _ACTION_MAP = _ACTION_MAPS["cpp"]

    def __init__(self):
        self._backend   = None
        self._game_text = ""
        self._title     = "PuzzleScript"
        self._level_i   = 0

    def _ensure_backend(self):
        if self._backend is None:
            from puzzlescript_cpp import CppPuzzleScriptBackend
            print("[CPP] Loading C++ engine…")
            self._backend = CppPuzzleScriptBackend()

    def load_game(self, game: str, level_i: int) -> None:
        from puzzlescript_jax.preprocessing import SIMPLIFIED_GAMES_DIR

        self._ensure_backend()
        print(f'[CPP] Compiling game: "{game}"')

        simp_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}_simplified.txt")
        if os.path.isfile(simp_path):
            # Simplified file already exists — use it directly so we avoid
            # the SIGALRM-based timeout that can't run in a background thread.
            with open(simp_path, "r", encoding="utf-8") as f:
                self._game_text = f.read()
            # CPP still needs the JS engine to compile → JSON → C++ load.
            self._backend._ensure_js_engine()
            self._backend._js_engine.compile(["restart"], self._game_text)
            json_str = str(self._backend._js_engine.serializeCompiledStateJSON())
            self._backend.cpp_engine.load_from_json(json_str)
            renderer = self._backend._ensure_renderer()
            renderer.load_sprite_data(
                str(self._backend._js_engine.serializeSpriteDataJSON()))
            renderer.load_render_config(json_str)
        else:
            from lark import Lark
            from puzzlescript_jax.globals import LARK_SYNTAX_PATH
            with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
                grammar = f.read()
            parser = Lark(grammar, start="ps_game", maybe_placeholders=False)
            self._game_text = self._backend.compile_game(parser, game)

        self._level_i  = level_i
        self._title    = game
        self._backend.load_level(self._game_text, level_i)
        self.game_text = self._game_text

    def render_frame(self) -> np.ndarray:
        return self._backend.render_frame()  # already RGB (H, W, 3)

    def step(self, direction: str) -> None:
        action_i = self._ACTION_MAP[direction]
        print(f"\n========= STEP =========")
        print(f"[CPP] action: {direction} (idx {action_i})")
        self._backend.cpp_engine.process_input(action_i)
        again_steps = 0
        while self._backend.cpp_engine.againing and again_steps < MAX_AGAIN:
            self._backend.cpp_engine.process_input(-1)
            again_steps += 1
        winning = self._backend.cpp_engine.winning
        score   = float(self._backend.cpp_engine.get_score())
        score_n = float(self._backend.cpp_engine.get_score_normalized())
        print(f"win={winning}  score={score:.3f}  score_norm={score_n:.3f}")

    def reset_level(self) -> None:
        self._backend.load_level(self._game_text, self._level_i)

    def load_level(self, level_i: int) -> None:
        self._level_i = level_i
        self._backend.load_level(self._game_text, level_i)

    def backup(self) -> Any:
        return self._backend.cpp_engine.backup_level()

    def restore(self, snapshot: Any) -> None:
        self._backend.cpp_engine.restore_level(snapshot)

    @property
    def winning(self) -> bool:
        return self._backend.cpp_engine.winning

    @property
    def restarting(self) -> bool:
        return False  # CPP doesn't surface a restart flag

    @property
    def num_levels(self) -> int:
        return self._backend.get_num_levels()

    @property
    def title(self) -> str:
        return self._title


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_BACKEND_NAMES = ("jax", "nodejs", "cpp")

def _make_backend(name: str, jit: bool = False, debug: bool = False,
                  profile: bool = False) -> PlayerBackend:
    name = name.lower()
    if name == "jax":
        return JaxPlayerBackend(jit=jit, debug=debug, profile=profile)
    elif name == "nodejs":
        return NodeJSPlayerBackend()
    elif name == "cpp":
        return CppPlayerBackend()
    raise ValueError(f"Unknown backend: {name!r}. Choose from {_BACKEND_NAMES}")


# ---------------------------------------------------------------------------
# Game session (backend-agnostic state)
# ---------------------------------------------------------------------------

class GameSession:
    def __init__(self, game: str, level: int, backend_name: str,
                 jit: bool, debug: bool, profile: bool):
        self.game        = game
        self.level_i     = level
        self.backend_name = backend_name
        self._jit        = jit
        self._debug      = debug
        self._profile    = profile
        self.lock        = threading.Lock()
        self._hist: list = []
        self._loading    = False
        # Cache of initialized backends — each keeps its own game state.
        self._backends: dict[str, PlayerBackend] = {}

        be = _make_backend(backend_name, jit=jit, debug=debug, profile=profile)
        self._backends[backend_name] = be
        self.backend = be
        self._load_game(game, level)

    # ------------------------------------------------------------------
    def _load_game(self, game: str, level_i: int):
        self._loading = True
        self.backend.load_game(game, level_i)
        self._hist = [self.backend.backup()]
        self.level_i = level_i
        self._loading = False

    def switch_backend(self, new_backend: str) -> dict:
        """Switch to a different backend, initializing it only the first time."""
        with self.lock:
            self._loading = True
            socketio.emit("backend_status",
                          {"loading": True, "backend": new_backend},
                          to=None)
            try:
                if new_backend in self._backends:
                    # Already initialized — just swap; preserve that backend's state.
                    self.backend      = self._backends[new_backend]
                    self.backend_name = new_backend
                    self._hist        = [self.backend.backup()]
                else:
                    new_be = _make_backend(new_backend, jit=self._jit,
                                           debug=self._debug,
                                           profile=self._profile)
                    new_be.load_game(self.game, self.level_i)
                    self._backends[new_backend] = new_be
                    self.backend      = new_be
                    self.backend_name = new_backend
                    self._hist        = [self.backend.backup()]
            finally:
                self._loading = False
            return self._frame_payload()

    # ------------------------------------------------------------------
    def render_frame_png_b64(self) -> str:
        im = self.backend.render_frame()
        h, w = im.shape[:2]
        im = cv2.resize(im, (w * SCALING_FACTOR, h * SCALING_FACTOR),
                        interpolation=cv2.INTER_NEAREST)
        buf = io.BytesIO()
        Image.fromarray(im).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _frame_payload(self) -> dict:
        return {
            "img":     self.render_frame_png_b64(),
            "status":  self._status_text(),
            "title":   self.backend.title,
            "backend": self.backend_name,
            "n_levels": self.backend.num_levels,
            "level_i": self.level_i,
        }

    def _status_text(self) -> str:
        if self.backend.winning:
            return f"Level {self.level_i} — YOU WIN!"
        return f"Level {self.level_i} / {self.backend.num_levels - 1}"

    # ------------------------------------------------------------------
    def handle_key(self, key: str) -> dict | None:
        with self.lock:
            if self._loading:
                return self._frame_payload()

            do_reset = False

            if key == "esc":
                return None  # signal quit

            elif key in ("left", "right", "up", "down", "x"):
                self.backend.step(key)
                self._hist.append(self.backend.backup())

                # auto-advance level on win
                if self.backend.winning:
                    next_i = self.level_i + 1
                    if next_i < self.backend.num_levels:
                        self.level_i = next_i
                        self.backend.load_level(next_i)
                        self._hist = [self.backend.backup()]
                elif getattr(self.backend, "restarting", False):
                    do_reset = True

            elif key == "r":
                do_reset = True

            elif key == "n":
                new_i = self.level_i + 1
                if new_i < self.backend.num_levels:
                    self.level_i = new_i
                    self.backend.load_level(new_i)
                    self._hist = [self.backend.backup()]

            elif key == "b":
                new_i = max(0, self.level_i - 1)
                self.level_i = new_i
                self.backend.load_level(new_i)
                self._hist = [self.backend.backup()]

            elif key == "z":
                if len(self._hist) > 1:
                    self._hist.pop()
                    self.backend.restore(self._hist[-1])

            if do_reset:
                self.backend.reset_level()
                self._hist = [self.backend.backup()]

            return self._frame_payload()


# ---------------------------------------------------------------------------
# HTML client
# ---------------------------------------------------------------------------

_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{{ title }}</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    background: #111; color: #eee; font-family: monospace;
    display: flex; flex-direction: column; align-items: center;
    padding: 1rem; gap: .5rem; margin: 0;
  }
  h1 { margin: 0; font-size: 1.2rem; }
  #meta { display:flex; gap:1rem; align-items:center; flex-wrap:wrap;
          justify-content:center; }
  #status { font-size:.85rem; color:#aef; }
  #hint   { font-size:.72rem; color:#666; text-align:center; }
  /* backend toggle */
  .be-btn {
    padding: .25rem .6rem; border-radius: 4px; border: 1px solid #555;
    background: #222; color: #aaa; cursor: pointer; font: inherit;
    font-size: .8rem; transition: background .15s, color .15s;
  }
  .be-btn:hover  { background: #333; color: #eee; }
  .be-btn.active { background: #2a5; color: #fff; border-color: #2a5; }
  .be-btn:disabled { opacity: .4; cursor: default; }
  /* game canvas */
  canvas {
    image-rendering: pixelated;
    border: 2px solid #333;
    max-width: 90vw; max-height: 60vh;
  }
  /* console */
  #console-wrap {
    width: min(90vw, 800px);
    display: flex; flex-direction: column; gap: .25rem;
  }
  #console-hdr {
    display: flex; justify-content: space-between; align-items: center;
    font-size: .75rem; color: #777;
  }
  #console {
    background: #0a0a0a; border: 1px solid #333; border-radius: 4px;
    height: 160px; overflow-y: auto; padding: .4rem .6rem;
    font-size: .72rem; line-height: 1.4; white-space: pre-wrap;
    word-break: break-all; color: #8c8;
  }
</style>
</head>
<body>

<h1 id="title">{{ title }}</h1>

<div id="meta">
  <div id="backend-btns">
    <button class="be-btn" data-be="jax">JAX</button>
    <button class="be-btn" data-be="nodejs">NodeJS</button>
    <button class="be-btn" data-be="cpp">CPP</button>
  </div>
  <div id="status">Connecting…</div>
</div>

<div id="hint">
  WASD / ↑↓←→ = move &nbsp;|&nbsp; X = action &nbsp;|&nbsp;
  R = restart &nbsp;|&nbsp; N = next &nbsp;|&nbsp; B = back &nbsp;|&nbsp;
  Z = undo &nbsp;|&nbsp; ESC = quit
</div>

<canvas id="canvas"></canvas>

<div id="console-wrap">
  <div id="console-hdr">
    <span>server console</span>
    <button onclick="clearConsole()" style="background:none;border:none;color:#555;cursor:pointer;font:inherit;font-size:.72rem;">clear</button>
  </div>
  <div id="console"></div>
</div>

<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
const socket  = io();
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const status  = document.getElementById('status');
const titleEl = document.getElementById('title');
const consoleEl = document.getElementById('console');
let currentBackend = '{{ backend }}';

// ── frame updates ──────────────────────────────────────────────────────────
socket.on('connect', () => { status.textContent = 'Connected'; });
socket.on('disconnect', () => { status.textContent = 'Disconnected'; });

socket.on('frame', data => {
  const img = new Image();
  img.onload = () => {
    canvas.width  = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
  img.src = 'data:image/png;base64,' + data.img;
  status.textContent = data.status;
  titleEl.textContent = data.title;
  if (data.backend) setActiveBackend(data.backend);
});

socket.on('backend_status', data => {
  if (data.loading) {
    status.textContent = `Loading ${data.backend} backend…`;
    document.querySelectorAll('.be-btn').forEach(b => b.disabled = true);
  } else {
    document.querySelectorAll('.be-btn').forEach(b => b.disabled = false);
    setActiveBackend(data.backend);
  }
});

// ── console ────────────────────────────────────────────────────────────────
socket.on('log', data => {
  consoleEl.textContent += data.text;
  consoleEl.scrollTop = consoleEl.scrollHeight;
});
function clearConsole() { consoleEl.textContent = ''; }

// ── backend toggle ─────────────────────────────────────────────────────────
function setActiveBackend(name) {
  currentBackend = name;
  document.querySelectorAll('.be-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.be === name);
  });
}

document.querySelectorAll('.be-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const be = btn.dataset.be;
    if (be === currentBackend) return;
    document.querySelectorAll('.be-btn').forEach(b => b.disabled = true);
    socket.emit('switch_backend', { backend: be });
  });
});

setActiveBackend('{{ backend }}');

// ── keyboard ───────────────────────────────────────────────────────────────
const KEY_MAP = {
  ArrowLeft:'left', a:'left',
  ArrowRight:'right', d:'right',
  ArrowUp:'up',    w:'up',
  ArrowDown:'down', s:'down',
  x:'x', X:'x',
  r:'r', R:'r',
  n:'n', N:'n',
  b:'b', B:'b',
  z:'z', Z:'z',
  Escape:'esc',
};

document.addEventListener('keydown', e => {
  const action = KEY_MAP[e.key];
  if (!action) return;
  e.preventDefault();
  socket.emit('keypress', { key: action });
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask / SocketIO routes
# ---------------------------------------------------------------------------

_session: Optional[GameSession] = None   # set before socketio.run()
_init_backend: str = "jax"


@app.route("/")
def index():
    title   = _session.backend.title if _session else "PuzzleScript"
    backend = _session.backend_name  if _session else _init_backend
    return render_template_string(_HTML, title=title, backend=backend)


@socketio.on("connect")
def on_connect():
    if _session is None:
        emit("frame", {"img": "", "status": "No game loaded",
                       "title": "Error", "backend": _init_backend})
        return
    emit("frame", _session._frame_payload())


@socketio.on("keypress")
def on_keypress(data):
    if _session is None:
        return
    key = data.get("key", "")
    payload = _session.handle_key(key)
    if payload is None:
        # ESC — tell client then shut down
        emit("frame", {"img": "", "status": "Bye!",
                       "title": _session.backend.title,
                       "backend": _session.backend_name})
        threading.Thread(target=socketio.stop, daemon=True).start()
    else:
        emit("frame", payload)


@socketio.on("switch_backend")
def on_switch_backend(data):
    if _session is None:
        return
    new_be = data.get("backend", "jax").lower()
    if new_be not in _BACKEND_NAMES:
        emit("backend_status", {"loading": False, "backend": _session.backend_name})
        return

    def _do_switch():
        try:
            payload = _session.switch_backend(new_be)
            socketio.emit("backend_status",
                          {"loading": False, "backend": new_be},
                          to=None)
            socketio.emit("frame", payload, to=None)
        except Exception as exc:
            print(f"[server] Backend switch failed: {exc}", file=sys.__stderr__)
            socketio.emit("backend_status",
                          {"loading": False, "backend": _session.backend_name},
                          to=None)

    # Run in background so the HTTP event loop isn't blocked
    threading.Thread(target=_do_switch, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def serve_game(game: str, level: int = 0, jit: bool = False,
               profile: bool = False, debug: bool = False,
               backend: str = "jax", host: str = "0.0.0.0",
               port: int = 5000):
    global _session, _init_backend

    _install_log_capture()
    _init_backend = backend

    print(f"Starting session: game={game!r}, level={level}, backend={backend}")
    _session = GameSession(game, level, backend,
                           jit=jit, debug=debug, profile=profile)

    print(f"\n  Open  http://<this-machine>:{port}/  in your local browser\n")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)


@hydra.main(config_name="config", version_base="1.3")
def main(cfg: Config):
    if cfg.game is not None:
        serve_game(cfg.game, level=cfg.level, jit=cfg.jit,
                   profile=cfg.profile, debug=cfg.debug,
                   backend=cfg.backend, host=cfg.host, port=cfg.port)
    else:
        from puzzlescript_jax.globals import TREES_DIR, TEST_GAMES
        tree_paths = sorted(glob.glob(os.path.join(TREES_DIR, "*")), reverse=True)
        test_paths = [os.path.join(TREES_DIR, tg + ".pkl") for tg in TEST_GAMES]
        all_paths  = test_paths + tree_paths
        games      = [os.path.basename(p)[:-4] for p in all_paths]
        if games:
            serve_game(games[0], jit=cfg.jit, debug=cfg.debug,
                       backend=cfg.backend, host=cfg.host, port=cfg.port)
        else:
            print("No games found. Pass game=<name> on the command line.")


if __name__ == "__main__":
    import jax
    jax.config.update("jax_platform_name", "cpu")
    main()
