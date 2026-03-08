from __future__ import annotations

import os
import random
from timeit import default_timer as timer
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from javascript import require

from puzzlejax.backends.base import PuzzleScriptSearchBackend, SearchResult
from puzzlejax.utils import level_to_int_arr
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
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    _ENGINE_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")
    _SOLVER_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "solver.js")

    def __init__(self) -> None:
        self.engine = require(self._ENGINE_PATH)
        self.solver = require(self._SOLVER_PATH)

    def compile_game(self, parser: Any, game: str) -> str:
        return compile_game(parser, self.engine, game, 0)

    def get_num_levels(self) -> int:
        return self.engine.getNumLevels()

    def load_level(self, game_text: str, level_i: int) -> None:
        self.engine.compile(["loadLevel", level_i], game_text)

    def unload_game(self) -> None:
        self.engine.unloadGame()

    def render_frame(self, level) -> np.ndarray:
        state = self.engine.getState()
        n_objs = len(list(state["idDict"]))
        level_arr = level_to_int_arr(level, n_objs)
        sprites = self._get_sprite_specs(state)
        visible_bounds = self._get_visible_bounds(level)
        mini, minj, maxi, maxj = visible_bounds
        cell_h, cell_w = sprites[0]["pixels"].shape[:2]
        frame = np.zeros(((maxj - minj) * cell_h, (maxi - mini) * cell_w, 3), dtype=np.uint8)
        frame[:] = self._hex_to_rgb(state["bgcolor"])

        for x in range(mini, maxi):
            for y in range(minj, maxj):
                cell_bits = int(level_arr[x, y])
                if cell_bits == 0:
                    continue
                px = (x - mini) * cell_w
                py = (y - minj) * cell_h
                for obj_id, sprite in enumerate(sprites):
                    if ((cell_bits >> obj_id) & 1) == 0:
                        continue
                    sprite_pixels = sprite["pixels"]
                    alpha_mask = sprite["mask"]
                    tile = frame[py:py + cell_h, px:px + cell_w]
                    tile[alpha_mask] = sprite_pixels[alpha_mask]
        return frame

    def render_gif(
        self,
        *,
        game_text: str,
        level_i: int,
        actions: list[int] | tuple[int, ...],
        gif_path: str,
        frame_duration_s: float = 0.1,
        scale: int = 10,
    ) -> str:
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

        gif_dir = os.path.dirname(gif_path)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
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

    @classmethod
    def _get_sprite_specs(cls, state) -> list[dict[str, np.ndarray]]:
        sprites = []
        for obj_name in list(state["idDict"]):
            obj = state["objects"][obj_name]
            sprite_rows = np.array([list(row) for row in obj["spritematrix"]], dtype=np.int16)
            mask = sprite_rows >= 0
            pixels = np.zeros((*sprite_rows.shape, 3), dtype=np.uint8)
            colors = [cls._hex_to_rgb(color) for color in list(obj["colors"])]
            for color_idx, color in enumerate(colors):
                pixels[sprite_rows == color_idx] = color
            sprites.append({"pixels": pixels, "mask": mask})
        return sprites

    @staticmethod
    def _hex_to_rgb(value: str) -> np.ndarray:
        value = value.lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) != 6:
            raise ValueError(f"Unsupported color value: {value!r}")
        return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
