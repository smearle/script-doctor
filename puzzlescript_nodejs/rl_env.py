from __future__ import annotations

from dataclasses import dataclass
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
