from __future__ import annotations

import base64
import os
import random
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import imageio
import numpy as np
from javascript import require

from backends.base import PuzzleScriptSearchBackend, SearchResult
from puzzlescript_jax.utils import level_to_int_arr
from puzzlescript_nodejs.utils import compile_game


class NodeJSPuzzleScriptBackend(PuzzleScriptSearchBackend):
    ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN", "ACTION"]
    MAX_ACTION_ID = len(ACTIONS) - 1
    SEARCH_ALGOS = {
        "bfs": "solveBFS",
        "astar": "solveAStar",
        "gbfs": "solveGBFS",
        "mcts": "solveMCTS",
        "random": "randomRollout",
        "python_random": "pythonRandomRollout",
    }
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    _ENGINE_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")
    _SOLVER_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "solver.js")
    _GIF_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "gif.js")

    def __init__(self) -> None:
        # createFreshApi() creates an isolated VM sandbox so that multiple
        # NodeJSPuzzleScriptBackend (or CppPuzzleScriptBackend) instances don't
        # share a single compiled-game state through the Node require() cache.
        _mod = require(self._ENGINE_PATH)
        self.engine = _mod.createFreshApi()
        self.solver = require(self._SOLVER_PATH)
        self.gif = require(self._GIF_PATH)

    def compile_game(self, parser: Any, game: str) -> str:
        return compile_game(parser, self.engine, game, 0)

    def get_num_levels(self) -> int:
        return self.engine.getNumLevels()

    def load_level(self, game_text: str, level_i: int) -> None:
        self.engine.compile(["loadLevel", level_i], game_text)

    def unload_game(self) -> None:
        self.engine.unloadGame()

    @staticmethod
    def _decode_frame(result) -> np.ndarray:
        """Decode a JS renderFrame result into an (H, W, 3) uint8 numpy array.

        The JS side returns ``dataBase64`` (a base64-encoded RGB byte string),
        so we need only one bridge crossing to transfer all pixel data instead
        of one crossing per pixel.
        """
        width  = int(result["width"])
        height = int(result["height"])
        raw = base64.b64decode(str(result["dataBase64"]))
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

    def render_frame(self, level=None) -> np.ndarray:
        """Render current (or given) level state using the JS-native renderer.

        ``level`` can be a JS level object (from backupLevel / takeAction) or
        ``None`` to render the current engine level.
        """
        result = self.engine.renderFrame(level) if level is not None else self.engine.renderFrame()
        if result is None:
            raise RuntimeError("JS renderFrame returned null -- is a game compiled?")
        return self._decode_frame(result)

    def render_frame_from_objects(
        self, objects: list[int], grid_w: int, grid_h: int,
    ) -> np.ndarray:
        """Render a frame from a raw objects array via JS-native renderer."""
        result = self.engine.renderFrameFromObjects(objects, grid_w, grid_h)
        if result is None:
            raise RuntimeError("JS renderFrameFromObjects returned null")
        return self._decode_frame(result)

    def render_frame_from_multihot_obs(self, obs: np.ndarray) -> np.ndarray:
        """Render a frame from a ``(C, H, W)`` multihot observation."""
        obs = np.asarray(obs)
        if obs.ndim != 3:
            raise ValueError(f"Expected obs with shape (C, H, W), got {obs.shape}")

        n_objs, grid_h, grid_w = (int(v) for v in obs.shape)
        stride_obj = (n_objs + 31) // 32
        objects = [0] * (grid_w * grid_h * stride_obj)

        for x in range(grid_w):
            for y in range(grid_h):
                flat_idx = (x * grid_h + y) * stride_obj
                for obj_i in range(n_objs):
                    if obs[obj_i, y, x]:
                        word = obj_i // 32
                        bit = obj_i % 32
                        objects[flat_idx + word] |= (1 << bit)

        return self.render_frame_from_objects(objects, grid_w, grid_h)

    def render_gif(
        self,
        *,
        game_text: str,
        level_i: int,
        actions: list[int] | tuple[int, ...],
        gif_path: str,
        frame_duration_s: float = 0.05,
        scale: int = 1,
    ) -> str:
        gif_dir = os.path.dirname(gif_path)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
        try:
            return self.gif.renderSolutionGif(
                {
                    "gameText": game_text,
                    "levelIndex": int(level_i),
                    "actions": [int(action) for action in actions],
                    "gifPath": gif_path,
                    "frameDurationMs": int(round(frame_duration_s * 1000)),
                    "scale": int(scale),
                }
            )
        except Exception:
            # Fallback preserves existing behavior if the JS-native path breaks.
            self.load_level(game_text, level_i)
            self.solver.precalcDistances(self.engine)
            frames = [self.render_frame(self.engine.backupLevel())]
            for action in actions:
                _, _, _, _, _, level, _, _ = self.solver.takeAction(self.engine, action)
                frames.append(self.render_frame(level))

            if scale > 1:
                frames = [
                    np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
                    for frame in frames
                ]

            imageio.mimsave(gif_path, frames, duration=frame_duration_s, loop=0)
            return gif_path

    def run_search(
        self,
        algo: str,
        *,
        game_text: str,
        level_i: int,
        n_steps: int,
        timeout_ms: int,
        warmup: bool = False,
    ) -> SearchResult:
        self.load_level(game_text, level_i)
        loops = 3 if warmup else 1

        if algo == "python_random":
            self.solver.precalcDistances(self.engine)
            raw_result = self._rand_rollout_from_python(n_steps=n_steps, timeout_ms=timeout_ms)
        else:
            method_name = self.SEARCH_ALGOS[algo]
            method = getattr(self.solver, method_name)
            raw_result = None
            for _ in range(loops):
                raw_result = method(
                    self.engine,
                    n_steps,
                    timeout_ms,
                    timeout=timeout_ms * 1.5 if timeout_ms > 0 else None,
                )

        return self._normalize_result(raw_result)

    def run_random_rollout_raw(
        self,
        *,
        game_text: str,
        level_i: int,
        n_steps: int,
        timeout_ms: int,
    ) -> dict[str, float | int | bool]:
        self.load_level(game_text, level_i)
        start_time = timer()
        completed_steps = 0
        timeout = False

        for completed_steps in range(n_steps):
            if timeout_ms > 0 and completed_steps % 1_000 == 0:
                if (timer() - start_time) * 1_000 > timeout_ms:
                    timeout = True
                    break

            action = random.randint(0, self.MAX_ACTION_ID)
            self.engine.processInput(action)
            while self.engine.getAgaining():
                self.engine.processInput(-1)

            if self.engine.getWinning():
                self._restart_after_win()
        else:
            completed_steps = n_steps

        elapsed = timer() - start_time
        return {
            "iterations": completed_steps,
            "time": elapsed,
            "timeout": timeout,
        }

    def _rand_rollout_from_python(self, *, n_steps: int, timeout_ms: int):
        start_time = timer()
        score = 0
        state = ""
        objects = []
        i = -1
        for i in range(n_steps):
            if timeout_ms > 0 and i % 1_000 == 0 and (timer() - start_time) * 1_000 > timeout_ms:
                elapsed = timer() - start_time
                return False, [], i, elapsed, score, state, False, list(objects)
            action = random.randint(0, self.MAX_ACTION_ID)
            _, _, _, _, score, state, _, objects = self.solver.takeAction(self.engine, action)
        return False, [], i, timer() - start_time, score, state, False, list(objects)

    def _restart_after_win(self) -> None:
        self.engine.clearBackups()
        if self.engine.getRestarting():
            return

        metadata = self.engine.getState().metadata
        if "norestart" in metadata:
            return

        self.engine.setRestarting(True)
        self.engine.addUndoState(self.engine.backupLevel())
        self.engine.restoreLevel(self.engine.getRestartTarget())

        if "run_rules_on_level_start" in metadata:
            self.engine.processInput(-1, True)

        level = self.engine.getLevel()
        level.commandQueue = []
        level.commandQueueSourceRules = []
        self.engine.setRestarting(False)

    @staticmethod
    def _normalize_result(raw_result) -> SearchResult:
        n_objs = len(list(raw_result[7]))
        end_level_state = level_to_int_arr(raw_result[5], n_objs).tolist()
        return SearchResult(
            solved=raw_result[0],
            actions=tuple(raw_result[1]),
            iterations=raw_result[2],
            time=raw_result[3],
            score=raw_result[4],
            state=end_level_state,
            timeout=raw_result[6],
            objs=list(raw_result[7]),
        )

    @staticmethod
    def _get_visible_bounds(level) -> tuple[int, int, int, int]:
        width = int(level["width"])
        height = int(level["height"])
        raw_bounds = list(level["oldflickscreendat"]) if "oldflickscreendat" in level else []
        if len(raw_bounds) != 4:
            return 0, 0, width, height
        mini, minj, maxi, maxj = (int(v) for v in raw_bounds)
        if mini >= maxi or minj >= maxj:
            return 0, 0, width, height
        return mini, minj, min(maxi, width), min(maxj, height)
