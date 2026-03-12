"""Python wrapper around the C++ PuzzleScript engine.

This module provides:
  - CppPuzzleScriptEngine: low-level engine API
  - CppPuzzleScriptBackend: high-level backend matching NodeJSPuzzleScriptBackend
  - CppPuzzleScriptEnv: single-env gym-style interface
  - CppBatchedPuzzleScriptEnv: batched/vectorized gym-style interface
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from backends.base import SearchResult
from puzzlescript_cpp._puzzlescript_cpp import Engine as _CppEngine, LevelBackup
from puzzlescript_cpp._puzzlescript_cpp import BatchedEngine as _BatchedEngine
from puzzlescript_cpp._puzzlescript_cpp import Renderer
from puzzlescript_cpp._puzzlescript_cpp import (
    MCTSOptions,
    RandomRolloutResult,
    SolverResult,
    random_rollout_raw as _random_rollout_raw,
    solve_astar as _solve_astar,
    solve_bfs as _solve_bfs,
    solve_gbfs as _solve_gbfs,
    solve_mcts as _solve_mcts,
    solve_random as _solve_random,
)


class CppPuzzleScriptEngine:
    """Low-level wrapper around the C++ PuzzleScript engine."""

    def __init__(self):
        self._engine = _CppEngine()
        self._json_state = None

    def load_from_json_file(self, path: str) -> None:
        with open(path, 'r') as f:
            json_str = f.read()
        self.load_from_json(json_str)

    def load_from_json(self, json_str: str) -> None:
        if not self._engine.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled state from JSON")
        self._json_state = json.loads(json_str)

    def load_level(self, level_index: int) -> None:
        self._engine.load_level(level_index)

    def process_input(self, direction: int) -> bool:
        """Process input. dir: 0=up, 1=left, 2=down, 3=right, 4=action, -1=tick."""
        return self._engine.process_input(direction)

    def check_win(self) -> bool:
        return self._engine.check_win()

    def get_score(self) -> float:
        return float(self._engine.get_score())

    def get_score_normalized(self) -> float:
        return float(self._engine.get_score_normalized())

    @property
    def winning(self) -> bool:
        return self._engine.is_winning()

    @property
    def againing(self) -> bool:
        return self._engine.is_againing()

    def get_objects(self) -> np.ndarray:
        return np.array(self._engine.get_objects(), dtype=np.int32)

    def get_objects_2d(self) -> np.ndarray:
        return np.array(self._engine.get_objects_2d(), dtype=np.int32)

    @property
    def width(self) -> int:
        return self._engine.get_width()

    @property
    def height(self) -> int:
        return self._engine.get_height()

    @property
    def object_count(self) -> int:
        return self._engine.get_object_count()

    @property
    def num_levels(self) -> int:
        return self._engine.get_num_levels()

    @property
    def id_dict(self) -> list[str]:
        return self._engine.get_id_dict()

    def restart(self) -> None:
        self._engine.restart()

    def backup_level(self) -> LevelBackup:
        return self._engine.backup_level()

    def restore_level(self, backup: LevelBackup) -> None:
        self._engine.restore_level(backup)

    def random_rollout_raw(self, max_iters: int = 100_000, timeout_ms: int = -1) -> RandomRolloutResult:
        return _random_rollout_raw(self._engine, max_iters, timeout_ms)

    def solve_random(
        self,
        max_length: int = 100,
        max_iters: int = 100_000,
        timeout_ms: int = 60_000,
    ) -> SolverResult:
        return _solve_random(self._engine, max_length, max_iters, timeout_ms)

    def solve_bfs(self, max_iters: int = 100_000, timeout_ms: int = -1) -> SolverResult:
        return _solve_bfs(self._engine, max_iters, timeout_ms)

    def solve_astar(self, max_iters: int = 100_000, timeout_ms: int = -1) -> SolverResult:
        return _solve_astar(self._engine, max_iters, timeout_ms)

    def solve_gbfs(self, max_iters: int = 100_000, timeout_ms: int = -1) -> SolverResult:
        return _solve_gbfs(self._engine, max_iters, timeout_ms)

    def solve_mcts(self, options: MCTSOptions | None = None) -> SolverResult:
        if options is None:
            options = MCTSOptions()
        return _solve_mcts(self._engine, options)


class CppPuzzleScriptBackend:
    """High-level backend for the C++ PuzzleScript engine.
    
    Uses the JS compiler (via Node.js) to compile games, then serializes
    the compiled state to JSON and loads it into the C++ engine.
    """
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    _ENGINE_JS_PATH = str(_ROOT_DIR / "puzzlescript_nodejs" / "puzzlescript" / "engine.js")
    SEARCH_ALGOS = {
        "bfs": "solve_bfs",
        "astar": "solve_astar",
        "gbfs": "solve_gbfs",
        "mcts": "solve_mcts",
        "random": "solve_random",
    }

    def __init__(self):
        self.cpp_engine = CppPuzzleScriptEngine()
        self._js_engine = None
        self._renderer = None

    def _ensure_js_engine(self):
        if self._js_engine is None:
            from javascript import require
            self._js_engine = require(self._ENGINE_JS_PATH)

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = Renderer()
        return self._renderer

    def compile_game(self, parser: Any, game: str) -> str:
        """Compile a game using the JS compiler and load into C++ engine."""
        self._ensure_js_engine()
        from puzzlescript_nodejs.utils import compile_game
        game_text = compile_game(parser, self._js_engine, game, 0)
        # Serialize the compiled state from JS and load into C++
        json_str = str(self._js_engine.serializeCompiledStateJSON())
        self.cpp_engine.load_from_json(json_str)
        renderer = self._ensure_renderer()
        renderer.load_sprite_data(str(self._js_engine.serializeSpriteDataJSON()))
        renderer.load_render_config(json_str)
        return game_text

    def compile_and_serialize(self, parser: Any, game: str) -> str:
        """Compile a game and return the serialized JSON state."""
        self._ensure_js_engine()
        from puzzlescript_nodejs.utils import compile_game
        compile_game(parser, self._js_engine, game, 0)
        return str(self._js_engine.serializeCompiledStateJSON())

    def load_from_json(self, json_str: str) -> None:
        """Load a pre-compiled game state from JSON."""
        self.cpp_engine.load_from_json(json_str)
        renderer = self._ensure_renderer()
        renderer.load_render_config(json_str)

    def load_level(self, game_text: str, level_i: int) -> None:
        """Load a level. Note: game must already be compiled."""
        self.cpp_engine.load_level(level_i)
        renderer = self._ensure_renderer()
        renderer.reset_viewport(self.cpp_engine.width, self.cpp_engine.height)

    def unload_game(self) -> None:
        self.cpp_engine = CppPuzzleScriptEngine()
        self._renderer = None

    def get_num_levels(self) -> int:
        return self.cpp_engine.num_levels

    def render_frame(self) -> np.ndarray:
        renderer = self._ensure_renderer()
        if not renderer.ready():
            raise RuntimeError("Renderer sprite data is not loaded")
        return np.asarray(renderer.render_engine(self.cpp_engine._engine), dtype=np.uint8)

    def render_frame_from_objects(
        self, objects: list[int] | np.ndarray, grid_w: int, grid_h: int,
    ) -> np.ndarray:
        renderer = self._ensure_renderer()
        if not renderer.ready():
            raise RuntimeError("Renderer sprite data is not loaded")
        objects_arr = np.asarray(objects, dtype=np.int32)
        return np.asarray(
            renderer.render_objects(objects_arr, int(grid_w), int(grid_h), self.cpp_engine.object_count),
            dtype=np.uint8,
        )

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
        self.load_level(game_text, level_i)
        frames = [self.render_frame()]
        for action in actions:
            self.process_input(int(action))
            again_steps = 0
            while self.againing and again_steps < MAX_AGAIN:
                self.process_input(-1)
                again_steps += 1
            frames.append(self.render_frame())

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

    def process_input(self, direction: int) -> bool:
        return self.cpp_engine.process_input(direction)

    @property
    def winning(self) -> bool:
        return self.cpp_engine.winning

    @property
    def againing(self) -> bool:
        return self.cpp_engine.againing

    def get_state(self) -> np.ndarray:
        return self.cpp_engine.get_objects()

    def get_id_dict(self) -> list[str]:
        return self.cpp_engine.id_dict

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
        method_name = self.SEARCH_ALGOS[algo]
        method = getattr(self.cpp_engine, method_name)
        raw_result = None
        for _ in range(loops):
            if algo == "random":
                raw_result = method(max_iters=n_steps, timeout_ms=timeout_ms)
            elif algo == "mcts":
                options = MCTSOptions()
                options.max_iterations = n_steps
                raw_result = method(options=options)
            else:
                raw_result = method(max_iters=n_steps, timeout_ms=timeout_ms)
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
        result = self.cpp_engine.random_rollout_raw(max_iters=n_steps, timeout_ms=timeout_ms)
        return {
            "iterations": result.iterations,
            "time": result.time,
            "timeout": result.timeout,
        }

    @staticmethod
    def _normalize_result(raw_result: SolverResult) -> SearchResult:
        n_objs = len(raw_result.id_dict)
        stride_obj = (n_objs + 31) // 32
        objects = np.asarray(raw_result.state.dat, dtype=np.int32)
        state = objects.reshape((raw_result.state.width, raw_result.state.height, stride_obj)).tolist()
        return SearchResult(
            solved=bool(raw_result.won),
            actions=tuple(raw_result.actions),
            iterations=int(raw_result.iterations),
            time=float(raw_result.time),
            score=float(raw_result.score),
            state=state,
            timeout=bool(raw_result.timeout),
            objs=list(raw_result.id_dict),
        )


# ---------------------------------------------------------------------------
# Gym-style environments
# ---------------------------------------------------------------------------

MAX_AGAIN = 50


class CppPuzzleScriptEnv:
    """Single-environment gym-style wrapper around the C++ engine.

    Matches the interface of ``NodeJSPuzzleEnv`` / ``PuzzleJaxEnv``:
      - ``reset()  -> (obs, info)``
      - ``step(action) -> (obs, reward, done, truncated, info)``

    Observations are multihot uint8 arrays of shape ``(n_objs, height, width)``.
    Actions: 0=up, 1=left, 2=down, 3=right, 4=action.
    """

    def __init__(self, json_str: str, level_i: int = 0,
                 max_episode_steps: int = 200):
        self._engine = _CppEngine()
        if not self._engine.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled game JSON")
        self._json_str = json_str
        self._compiled = json.loads(json_str)
        self._level_i = level_i
        self._max_steps = max_episode_steps
        self._steps = 0
        self._max_height, self._max_width = self._compute_max_level_shape([level_i])

        # Load level once to get geometry
        probe_level_i = level_i if level_i >= 0 else 0
        self._engine.load_level(probe_level_i)
        self._n_objs = self._engine.get_object_count()
        self._width = self._max_width
        self._height = self._max_height
        self._stride_obj = (self._n_objs + 31) // 32
        self._prev_score = float(self._engine.get_score())

    def _compute_max_level_shape(self, level_indices: list[int] | None = None) -> tuple[int, int]:
        if level_indices is None or any(level_i < 0 for level_i in level_indices):
            allowed_indices = None
        else:
            allowed_indices = {int(level_i) for level_i in level_indices}
        max_width = 0
        max_height = 0
        for level in self._compiled.get("levels", []):
            if level.get("type") != "level":
                continue
            if allowed_indices is not None and int(level["index"]) not in allowed_indices:
                continue
            max_width = max(max_width, int(level["width"]))
            max_height = max(max_height, int(level["height"]))
        return max_height, max_width

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self._n_objs, self._height, self._width)

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def num_levels(self) -> int:
        return self._engine.get_num_levels()

    def set_level(self, level_i: int) -> None:
        self._level_i = level_i

    def reset(self) -> tuple[np.ndarray, dict]:
        if self._level_i < 0:
            self._level_i = int(np.random.randint(self.num_levels))
        self._engine.load_level(self._level_i)
        self._steps = 0
        self._prev_score = float(self._engine.get_score())
        obs = self._get_obs()
        return obs, {"won": False, "steps": 0, "score": self._prev_score, "score_delta": 0.0, "level_i": self._level_i}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._engine.process_input(action)
        ag = 0
        while self._engine.is_againing() and ag < MAX_AGAIN:
            self._engine.process_input(-1)
            ag += 1

        self._steps += 1
        won = self._engine.is_winning()
        truncated = self._steps >= self._max_steps
        done = won
        score = float(self._engine.get_score())
        score_delta = self._prev_score - score
        reward = score_delta + (1.0 if won else 0.0) - 0.01
        self._prev_score = score
        obs = self._get_obs()
        info = {"won": won, "steps": self._steps, "score": score, "score_delta": score_delta, "level_i": self._level_i}
        return obs, reward, done, truncated, info

    def get_objects(self) -> np.ndarray:
        return np.array(self._engine.get_objects(), dtype=np.int32)

    def _get_obs(self) -> np.ndarray:
        objects = self._engine.get_objects()
        width = self._engine.get_width()
        height = self._engine.get_height()
        stride_obj = len(objects) // (width * height)
        obs = np.zeros((self._n_objs, self._height, self._width), dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                flat_idx = (x * height + y) * stride_obj
                for obj in range(self._n_objs):
                    word = obj // 32
                    bit = obj % 32
                    if word < stride_obj and (int(objects[flat_idx + word]) & (1 << bit)):
                        obs[obj, y, x] = 1
        return obs


class CppBatchedPuzzleScriptEnv:
    """Batched/vectorized gym-style wrapper around the C++ BatchedEngine.

    Manages ``batch_size`` independent environments running in parallel.

    Interface:
      - ``reset()         -> obs``         (reset all)
      - ``reset(indices)  -> obs``         (reset specific envs)
      - ``step(actions)   -> (obs, rewards, dones, truncated, infos)``

    Observations: uint8 ``(batch, n_objs, height, width)``
    Actions: int32 ``(batch,)``, values 0-4.
    Auto-resets environments that reach ``done`` when ``auto_reset=True``.
    """

    def __init__(self, json_str: str, batch_size: int,
                 level_indices: list[int] | None = None,
                 max_episode_steps: int = 200,
                 auto_reset: bool = True):
        self._be = _BatchedEngine(batch_size)
        if not self._be.load_from_json(json_str):
            raise RuntimeError("Failed to load compiled game JSON")
        self._json_str = json_str
        self._compiled = json.loads(json_str)
        self._max_steps = max_episode_steps
        self._auto_reset = bool(auto_reset)
        if level_indices is None:
            level_indices = [0] * batch_size
        self._max_height, self._max_width = self._compute_max_level_shape(level_indices)
        self._n_objs = int(self._compiled["objectCount"])
        self._stride_obj = (self._n_objs + 31) // 32
        self._base_level_indices = np.asarray(level_indices, dtype=np.int32)
        self._active_level_indices = self._base_level_indices.copy()
        self._steps = np.zeros(batch_size, dtype=np.int32)
        self._be.set_levels(self._active_level_indices.tolist())
        if hasattr(self._be, "set_auto_reset"):
            self._be.set_auto_reset(False)

        # Do an initial reset to populate geometry caches.
        self.reset()

    @property
    def batch_size(self) -> int:
        return self._be.batch_size

    @property
    def num_objects(self) -> int:
        return self._be.num_objects

    @property
    def observation_shape(self) -> tuple[int, int, int, int]:
        """(batch, n_objs, height, width)"""
        return (self.batch_size, self._n_objs, self._max_height, self._max_width)

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def num_levels(self) -> int:
        return self._be.num_levels

    def set_levels(self, level_indices: list[int]) -> None:
        self._base_level_indices = np.asarray(level_indices, dtype=np.int32)
        self._active_level_indices = self._base_level_indices.copy()
        self._max_height, self._max_width = self._compute_max_level_shape(level_indices)
        self._be.set_levels(self._active_level_indices.tolist())

    def _compute_max_level_shape(self, level_indices: list[int] | None = None) -> tuple[int, int]:
        if level_indices is None or any(level_i < 0 for level_i in level_indices):
            allowed_indices = None
        else:
            allowed_indices = {int(level_i) for level_i in level_indices}
        max_width = 0
        max_height = 0
        for level in self._compiled.get("levels", []):
            if level.get("type") != "level":
                continue
            if allowed_indices is not None and int(level["index"]) not in allowed_indices:
                continue
            max_width = max(max_width, int(level["width"]))
            max_height = max(max_height, int(level["height"]))
        return max_height, max_width

    def _get_obs_for_env(self, env_idx: int) -> np.ndarray:
        objects = np.array(self._be.get_objects(env_idx), dtype=np.int32)
        width = int(self._be.get_width(env_idx))
        height = int(self._be.get_height(env_idx))
        stride_obj = len(objects) // (width * height)
        obs = np.zeros((self._n_objs, self._max_height, self._max_width), dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                flat_idx = (x * height + y) * stride_obj
                for obj in range(self._n_objs):
                    word = obj // 32
                    bit = obj % 32
                    if word < stride_obj and (int(objects[flat_idx + word]) & (1 << bit)):
                        obs[obj, y, x] = 1
        return obs

    def _get_obs_batch(self) -> np.ndarray:
        return np.stack([self._get_obs_for_env(i) for i in range(self.batch_size)], axis=0)

    def _sample_level_indices(self) -> np.ndarray:
        random_mask = self._base_level_indices < 0
        if not np.any(random_mask):
            return self._base_level_indices.copy()
        sampled = self._base_level_indices.copy()
        sampled[random_mask] = np.random.randint(self.num_levels, size=int(random_mask.sum()), dtype=np.int32)
        return sampled

    def _assign_levels_for_reset(self, env_indices: np.ndarray) -> None:
        if env_indices.size == 0:
            return
        random_mask = self._base_level_indices[env_indices] < 0
        if not np.any(random_mask):
            return
        sampled = np.random.randint(self.num_levels, size=int(random_mask.sum()), dtype=np.int32)
        self._active_level_indices[env_indices[random_mask]] = sampled
        self._be.set_levels(self._active_level_indices.tolist())

    def reset(self, env_indices: list[int] | None = None) -> np.ndarray:
        if env_indices is None:
            self._active_level_indices = self._sample_level_indices()
            self._be.set_levels(self._active_level_indices.tolist())
            self._be.reset_all()
            self._steps[:] = 0
        else:
            env_indices_arr = np.asarray(env_indices, dtype=np.int32)
            self._assign_levels_for_reset(env_indices_arr)
            self._be.reset(env_indices)
            for i in env_indices_arr.tolist():
                self._steps[i] = 0
        return self._get_obs_batch()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions = np.asarray(actions, dtype=np.int32)
        level_indices = self._active_level_indices.copy()
        self._be.step(actions)
        self._steps += 1

        obs = self._get_obs_batch()
        rewards = np.array(self._be.get_rewards())
        scores = np.array(self._be.get_scores())
        score_deltas = np.array(self._be.get_score_deltas())
        dones = np.array(self._be.get_dones())
        wins = np.array(self._be.get_wins())
        truncated = self._steps >= self._max_steps

        if self._auto_reset:
            reset_indices = np.where(dones | truncated)[0].tolist()
            if reset_indices:
                self.reset(reset_indices)
                obs = self._get_obs_batch()

        infos = {
            "won": wins,
            "steps": self._steps.copy(),
            "score": scores,
            "score_delta": score_deltas,
            "level_i": level_indices,
            "next_level_i": self._active_level_indices.copy(),
        }
        return obs, rewards, dones | truncated, truncated, infos

    def get_objects(self, env_idx: int) -> np.ndarray:
        return np.array(self._be.get_objects(env_idx))
