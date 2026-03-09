from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from puzzlejax.backends import NodeJSPuzzleScriptBackend
from puzzlejax.env import PSObs
from puzzlejax.utils import init_ps_lark_parser, level_to_int_arr


@dataclass
class NodeJSEnvState:
    score: float
    won: bool
    steps: int
    level: dict[str, Any]
    objs: list[str]


class NodeJSPuzzleEnv:
    """Small imperative RL wrapper around the original PuzzleScript engine."""

    def __init__(self, game: str, level_i: int, max_episode_steps: int) -> None:
        self.game = game
        self.level_i = level_i
        self.max_steps = max_episode_steps
        self.parser = init_ps_lark_parser()
        self.backend = NodeJSPuzzleScriptBackend()
        self.game_text = self.backend.compile_game(self.parser, game)
        self.action_space = spaces.Discrete(self.backend.MAX_ACTION_ID + 1)
        self._obs_shape = self._infer_obs_shape()

    def _infer_obs_shape(self) -> tuple[int, int, int]:
        self.backend.load_level(self.game_text, self.level_i)
        level = self.backend.engine.backupLevel()
        objs = list(self.backend.engine.getState().idDict)
        obs = self._obs_from_level(level, objs)
        return tuple(int(dim) for dim in obs.multihot_level.shape)

    def _obs_from_level(self, level: dict[str, Any], objs: list[str]) -> PSObs:
        int_level = level_to_int_arr(level, len(objs))
        multihot = ((int_level[..., None] & (1 << np.arange(len(objs)))) > 0).astype(np.float32)
        multihot = np.transpose(multihot, (2, 0, 1))
        return PSObs(multihot_level=jnp.asarray(multihot), flat_obs=None)

    def gen_dummy_obs(self, _params=None) -> PSObs:
        return PSObs(multihot_level=jnp.zeros((1,) + self._obs_shape, dtype=jnp.float32), flat_obs=None)

    def reset(self, seed: int | None = None) -> tuple[PSObs, NodeJSEnvState]:
        del seed
        self.backend.load_level(self.game_text, self.level_i)
        self.backend.solver.precalcDistances(self.backend.engine)
        level = self.backend.engine.backupLevel()
        objs = list(self.backend.engine.getState().idDict)
        score = float(self.backend.solver.getScore(self.backend.engine))
        state = NodeJSEnvState(score=score, won=bool(self.backend.engine.getWinning()), steps=0, level=level, objs=objs)
        return self._obs_from_level(level, objs), state

    def step(
        self, state: NodeJSEnvState, action: int
    ) -> tuple[PSObs, NodeJSEnvState, float, bool, dict[str, Any]]:
        _, _, _, _, score, level, _, objs = self.backend.solver.takeAction(self.backend.engine, int(action))
        won = bool(self.backend.engine.getWinning())
        next_steps = state.steps + 1
        reward = float(score) - float(state.score)
        if won:
            reward += 1.0
        reward -= 0.01
        done = won or next_steps >= self.max_steps
        next_state = NodeJSEnvState(
            score=float(score),
            won=won,
            steps=next_steps,
            level=level,
            objs=list(objs),
        )
        info = {
            "score": float(score),
            "won": won,
            "steps": next_steps,
        }
        return self._obs_from_level(level, next_state.objs), next_state, reward, done, info

    def render_gif(
        self,
        *,
        actions: list[int],
        gif_path: str,
        frame_duration_s: float,
        scale: int,
    ) -> str:
        return self.backend.render_gif(
            game_text=self.game_text,
            level_i=self.level_i,
            actions=actions,
            gif_path=gif_path,
            frame_duration_s=frame_duration_s,
            scale=scale,
        )


class _NodeJSBatchedController:
    _ROOT_DIR = Path(__file__).resolve().parent
    _CONTROLLER_PATH = _ROOT_DIR / "puzzlescript" / "batched_env_controller.js"

    def __init__(self, *, game_text: str, level_i: int, n_envs: int, max_episode_steps: int) -> None:
        self.n_envs = int(n_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.proc = subprocess.Popen(
            ["node", str(self._CONTROLLER_PATH)],
            cwd=os.getcwd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ready, obs = self._request(
            {
                "cmd": "init",
                "gameText": game_text,
                "levelI": int(level_i),
                "nEnvs": self.n_envs,
                "maxEpisodeSteps": self.max_episode_steps,
            },
            obs_shape=None,
        )
        self.width = int(ready["width"])
        self.height = int(ready["height"])
        self.object_count = int(ready["object_count"])
        self.object_names = list(ready["object_names"])
        self.obs_shape = (self.n_envs, self.object_count, self.width, self.height)
        self._last_obs = None if obs is None else obs.reshape(self.obs_shape)

    def _read_exact(self, n_bytes: int) -> bytes:
        assert self.proc.stdout is not None
        data = self.proc.stdout.read(n_bytes)
        if data is None or len(data) != n_bytes:
            stderr = b""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read() or b""
            raise RuntimeError(
                "Node batched controller returned incomplete binary payload. "
                f"stderr: {stderr.decode(errors='replace')}"
            )
        return data

    def _request(
        self,
        payload: dict[str, Any],
        *,
        obs_shape: tuple[int, ...] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Node batched controller pipes are unavailable.")

        self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

        header_raw = self.proc.stdout.readline()
        if not header_raw:
            stderr = b""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read() or b""
            raise RuntimeError(
                "Node batched controller exited without a response. "
                f"stderr: {stderr.decode(errors='replace')}"
            )

        header = json.loads(header_raw.decode("utf-8"))
        if not header.get("ok", False):
            raise RuntimeError(header.get("error", "Node batched controller request failed."))

        obs = None
        obs_bytes = int(header.get("obs_bytes", 0))
        if obs_bytes > 0:
            raw = self._read_exact(obs_bytes)
            target_shape = obs_shape or getattr(self, "obs_shape", None)
            if target_shape is None:
                obs = np.frombuffer(raw, dtype=np.uint8).copy()
            else:
                obs = np.frombuffer(raw, dtype=np.uint8).reshape(target_shape).copy()
        return header, obs

    def reset(self, env_indices: list[int] | None = None) -> np.ndarray:
        _, obs = self._request({"cmd": "reset", "indices": env_indices})
        assert obs is not None
        self._last_obs = obs
        return obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        header, obs = self._request(
            {
                "cmd": "step",
                "actions": np.asarray(actions, dtype=np.int32).tolist(),
            }
        )
        assert obs is not None
        self._last_obs = obs
        rewards = np.asarray(header["rewards"], dtype=np.float32)
        dones = np.asarray(header["dones"], dtype=bool)
        truncated = np.asarray(header["truncated"], dtype=bool)
        infos = {
            "won": np.asarray(header["won"], dtype=bool),
            "steps": np.asarray(header["steps"], dtype=np.int32),
            "score": np.asarray(header["score"], dtype=np.float32),
        }
        return obs, rewards, dones, truncated, infos

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self._request({"cmd": "close"})
        except Exception:
            self.proc.terminate()
        finally:
            self.proc.wait(timeout=5.0)


class NodeJSBatchedPuzzleEnv:
    """Vectorized NodeJS PuzzleScript environment backed by a Node-managed worker pool."""

    def __init__(self, game: str, level_i: int, batch_size: int, max_episode_steps: int) -> None:
        self.game = game
        self.level_i = level_i
        self.batch_size = int(batch_size)
        self.max_steps = int(max_episode_steps)
        self.parser = init_ps_lark_parser()
        self.backend = NodeJSPuzzleScriptBackend()
        self.game_text = self.backend.compile_game(self.parser, game)
        self.action_space = spaces.Discrete(self.backend.MAX_ACTION_ID + 1)
        self._controller = _NodeJSBatchedController(
            game_text=self.game_text,
            level_i=self.level_i,
            n_envs=self.batch_size,
            max_episode_steps=self.max_steps,
        )
        self._obs_shape = self._controller.obs_shape

    @property
    def num_actions(self) -> int:
        return self.backend.MAX_ACTION_ID + 1

    @property
    def observation_shape(self) -> tuple[int, int, int, int]:
        return self._obs_shape

    def reset(self, env_indices: list[int] | None = None) -> np.ndarray:
        return self._controller.reset(env_indices)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        return self._controller.step(actions)

    def gen_dummy_obs(self, _params=None) -> PSObs:
        return PSObs(multihot_level=jnp.zeros(self._obs_shape, dtype=jnp.float32), flat_obs=None)

    def as_ps_obs(self, obs: np.ndarray) -> PSObs:
        return PSObs(multihot_level=jnp.asarray(obs, dtype=jnp.float32), flat_obs=None)

    def close(self) -> None:
        self._controller.close()

    def __enter__(self) -> "NodeJSBatchedPuzzleEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()
