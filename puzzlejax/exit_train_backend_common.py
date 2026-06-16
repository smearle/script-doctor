from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import imageio
import numpy as np
from javascript import require

from backends import NodeJSPuzzleScriptBackend
from backends.base import SearchResult
from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv, Renderer
from puzzlescript_nodejs.rl_env import NodeJSPuzzleEnv
from puzzlescript_nodejs.utils import compile_game


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_JS_PATH = os.path.join(ROOT_DIR, "puzzlescript_nodejs", "puzzlescript", "engine.js")


@dataclass
class BackendRuntime:
    backend_name: str
    game: str
    level_i: int
    max_episode_steps: int
    game_text: str
    json_str: str
    sprite_json: str
    search_backend: object


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = int(max_size)
        self.obs: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.size = 0
        self.write_ptr = 0

    def _ensure_storage(self, obs: np.ndarray) -> None:
        if self.obs is not None:
            return
        self.obs = np.empty((self.max_size, *obs.shape[1:]), dtype=obs.dtype)
        self.actions = np.empty((self.max_size,), dtype=np.int64)
        self.values = np.empty((self.max_size,), dtype=np.float32)
        self.weights = np.empty((self.max_size,), dtype=np.float32)

    def add(self, batch: dict[str, np.ndarray]) -> None:
        obs = np.asarray(batch["obs"])
        actions = np.asarray(batch["actions"], dtype=np.int64)
        values = np.asarray(batch["values"], dtype=np.float32)
        weights = np.asarray(batch["weights"], dtype=np.float32)
        if obs.shape[0] == 0:
            return

        self._ensure_storage(obs)
        n = obs.shape[0]
        if n >= self.max_size:
            self.obs[...] = obs[-self.max_size :]
            self.actions[...] = actions[-self.max_size :]
            self.values[...] = values[-self.max_size :]
            self.weights[...] = weights[-self.max_size :]
            self.size = self.max_size
            self.write_ptr = 0
            return

        end_ptr = self.write_ptr + n
        if end_ptr <= self.max_size:
            self.obs[self.write_ptr:end_ptr] = obs
            self.actions[self.write_ptr:end_ptr] = actions
            self.values[self.write_ptr:end_ptr] = values
            self.weights[self.write_ptr:end_ptr] = weights
        else:
            first = self.max_size - self.write_ptr
            second = n - first
            self.obs[self.write_ptr:] = obs[:first]
            self.actions[self.write_ptr:] = actions[:first]
            self.values[self.write_ptr:] = values[:first]
            self.weights[self.write_ptr:] = weights[:first]
            self.obs[:second] = obs[first:]
            self.actions[:second] = actions[first:]
            self.values[:second] = values[first:]
            self.weights[:second] = weights[first:]

        self.write_ptr = (self.write_ptr + n) % self.max_size
        self.size = min(self.max_size, self.size + n)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError("Replay buffer is empty.")
        indices = rng.choice(self.size, size=min(batch_size, self.size), replace=False)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "values": self.values[indices],
            "weights": self.weights[indices],
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            np.savez_compressed(
                f,
                obs=self.obs,
                actions=self.actions,
                values=self.values,
                weights=self.weights,
                size=np.asarray(self.size, dtype=np.int64),
                write_ptr=np.asarray(self.write_ptr, dtype=np.int64),
                max_size=np.asarray(self.max_size, dtype=np.int64),
            )

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with np.load(path, allow_pickle=False) as data:
            self.obs = data["obs"] if "obs" in data else None
            self.actions = data["actions"] if "actions" in data else None
            self.values = data["values"] if "values" in data else None
            self.weights = data["weights"] if "weights" in data else None
            self.size = int(data["size"]) if "size" in data else 0
            self.write_ptr = int(data["write_ptr"]) if "write_ptr" in data else 0
        return self.obs is not None


def compile_game_json(game: str) -> tuple[str, str]:
    parser = init_ps_lark_parser()
    js_engine = require(ENGINE_JS_PATH)
    compile_game(parser, js_engine, game, 0)
    return str(js_engine.serializeCompiledStateJSON()), str(js_engine.serializeSpriteDataJSON())


def build_backend_runtime(
    *,
    game: str,
    level_i: int,
    backend_name: str,
    max_episode_steps: int,
) -> BackendRuntime:
    parser = init_ps_lark_parser()
    json_str, sprite_json = compile_game_json(game)

    if backend_name == "cpp":
        search_backend = CppPuzzleScriptBackend()
        game_text = search_backend.compile_game(parser, game)
    elif backend_name == "nodejs":
        search_backend = NodeJSPuzzleScriptBackend()
        game_text = search_backend.compile_game(parser, game)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

    return BackendRuntime(
        backend_name=backend_name,
        game=game,
        level_i=int(level_i),
        max_episode_steps=int(max_episode_steps),
        game_text=game_text,
        json_str=json_str,
        sprite_json=sprite_json,
        search_backend=search_backend,
    )


def make_env(runtime: BackendRuntime):
    if runtime.backend_name == "cpp":
        return CppPuzzleScriptEnv(
            runtime.json_str,
            level_i=runtime.level_i,
            max_episode_steps=runtime.max_episode_steps,
        )
    if runtime.backend_name == "nodejs":
        return NodeJSPuzzleEnv(
            runtime.game,
            runtime.level_i,
            runtime.max_episode_steps,
            game_text=runtime.game_text,
        )
    raise ValueError(f"Unsupported backend: {runtime.backend_name}")


def run_search(
    runtime: BackendRuntime,
    *,
    algo: str,
    n_steps: int,
    timeout_ms: int,
) -> SearchResult:
    return runtime.search_backend.run_search(
        algo,
        game_text=runtime.game_text,
        level_i=runtime.level_i,
        n_steps=n_steps,
        timeout_ms=timeout_ms,
        warmup=False,
    )


def collect_expert_examples(
    runtime: BackendRuntime,
    search_result: SearchResult,
    *,
    unsolved_value_bonus: float = 5.0,
) -> dict[str, np.ndarray | bool | float | int]:
    env = make_env(runtime)
    try:
        obs, _info = env.reset()
        obs_list = []
        action_list = []
        value_list = []
        weight_list = []
        episode_return = 0.0
        won = False

        actions = list(search_result.actions)
        for idx, action in enumerate(actions):
            obs_list.append(np.asarray(obs, dtype=np.uint8))
            action_list.append(int(action))
            remaining = float(len(actions) - idx)
            target_value = remaining if search_result.solved else remaining + unsolved_value_bonus
            value_list.append(target_value)
            weight_list.append(2.0 if search_result.solved else 0.5)

            obs, reward, done, truncated, info = env.step(int(action))
            episode_return += float(reward)
            won = bool(info.get("won", False))
            if done or truncated:
                break

        obs_arr = np.stack(obs_list, axis=0) if obs_list else np.empty((0, *np.asarray(obs).shape), dtype=np.uint8)
        return {
            "obs": obs_arr,
            "actions": np.asarray(action_list, dtype=np.int64),
            "values": np.asarray(value_list, dtype=np.float32),
            "weights": np.asarray(weight_list, dtype=np.float32),
            "solved": bool(search_result.solved),
            "expert_return": float(episode_return),
            "expert_won": bool(won),
            "trace_len": len(action_list),
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def save_rollout_gif(
    runtime: BackendRuntime,
    actions: list[int],
    gif_path: str,
    *,
    frame_duration_s: float,
    scale: int = 10,
) -> str:
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    if runtime.backend_name == "nodejs":
        env = make_env(runtime)
        try:
            return env.render_gif(
                actions=actions,
                gif_path=gif_path,
                frame_duration_s=frame_duration_s,
                scale=scale,
            )
        finally:
            if hasattr(env, "close"):
                env.close()

    renderer = Renderer()
    renderer.load_sprite_data(runtime.sprite_json)
    env = CppPuzzleScriptEnv(
        runtime.json_str,
        level_i=runtime.level_i,
        max_episode_steps=runtime.max_episode_steps,
    )
    try:
        obs, _info = env.reset()
        n_objs, h, w = env.observation_shape
        frames = [renderer.render_obs(obs, n_objs, h, w)]
        for action in actions:
            obs, _reward, done, truncated, _info = env.step(int(action))
            frames.append(renderer.render_obs(obs, n_objs, h, w))
            if done or truncated:
                break
        imageio.mimsave(gif_path, frames, duration=frame_duration_s, loop=0)
        return gif_path
    finally:
        if hasattr(env, "close"):
            env.close()


def write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
