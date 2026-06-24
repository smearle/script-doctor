"""Vectorized PuzzleScript env over the C++ batched engine, with a PufferLib /
gym-vector style API (batched reset/step returning numpy arrays). This is the
env-adapter layer for the PufferLib port (see PUFFERLIB_PORT_PLAN.md).

Design notes / why it lives here:
- The C++ engine (`CppBatchedPuzzleScriptEnv`) already steps B envs across CPU
  threads (OpenMP, GIL released) at ~280k steps/s, and computes reward/done. So
  this adapter is thin: it owns one batched engine per game, pads each game's
  multihot obs to a common (C,H,W), concatenates, and tracks a per-env game id.
- Single-game is the special case `games=[g]`.
- DEPENDENCY NOTE: pufferlib 3.0 hard-pins numpy<2 and gymnasium<=0.29, which
  conflicts with this repo's JAX stack (numpy>=2). Importing the
  `puzzlescript_cpp` *wrapper* also drags in `backends.nodejs` -> javascript/jax.
  So a true in-pufferlib run needs an isolated venv using the raw `_puzzlescript_cpp`
  .so. This adapter is validated in the main venv with torch (no pufferlib import
  needed); its API (single_observation_space / single_action_space / reset / step)
  matches what a PufferEnv wrapper expects.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import gymnasium as gym

from puzzlescript_cpp import CppPuzzleScriptBackend, CppBatchedPuzzleScriptEnv

JSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "puffer_assets", "games_json")


def _load_or_compile_json(game: str, json_dir: str = JSON_DIR) -> str:
    path = os.path.join(json_dir, f"{game}.json")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    # Fall back to compiling via the JS backend (needs Node; main venv only).
    from puzzlescript_jax.utils import init_ps_lark_parser
    os.makedirs(json_dir, exist_ok=True)
    js = CppPuzzleScriptBackend().compile_and_serialize(init_ps_lark_parser(), game)
    with open(path, "w") as f:
        f.write(js)
    return js


class PuzzleScriptVecEnv:
    """Batched, optionally multi-game PuzzleScript env.

    obs: uint8 (num_envs, C, H, W) multihot, padded to a common shape across
    games. actions: int32 (num_envs,) in [0,5). step() returns
    (obs, rewards, dones, truncations, infos). Auto-resets on done (C++ side).
    `game_ids` gives the per-env game index (for conditioning / logging).
    """

    def __init__(self, games: List[str], num_envs: int, max_episode_steps: int = 200,
                 level_indices: Optional[List[int]] = None, num_threads: int = 0,
                 json_dir: str = JSON_DIR):
        assert num_envs % len(games) == 0, "num_envs must be divisible by #games"
        self.games = list(games)
        self.N = len(self.games)
        self.epg = num_envs // self.N
        self.num_envs = self.epg * self.N

        self._engines = []
        shapes = []
        for g in self.games:
            js = _load_or_compile_json(g, json_dir)
            lvl = level_indices if level_indices is not None else [-1] * self.epg
            e = CppBatchedPuzzleScriptEnv(js, batch_size=self.epg, level_indices=lvl,
                                          max_episode_steps=max_episode_steps,
                                          auto_reset=True, num_threads=num_threads)
            self._engines.append(e)
            shapes.append(tuple(np.asarray(e.observation_shape)[1:]))  # (C,H,W)
        self.Cmax = max(s[0] for s in shapes)
        self.Hmax = max(s[1] for s in shapes)
        self.Wmax = max(s[2] for s in shapes)
        self._shapes = shapes
        self.game_ids = np.concatenate([np.full(self.epg, i, np.int32) for i in range(self.N)])

        self.single_observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.Cmax, self.Hmax, self.Wmax), dtype=np.uint8)
        self.single_action_space = gym.spaces.Discrete(int(self._engines[0].num_actions))

    def _pad(self, obs, ci, hi, wi):
        if (ci, hi, wi) == (self.Cmax, self.Hmax, self.Wmax):
            return obs
        out = np.zeros((obs.shape[0], self.Cmax, self.Hmax, self.Wmax), dtype=np.uint8)
        out[:, :ci, :hi, :wi] = obs
        return out

    def reset(self):
        obs_parts = []
        for gi, e in enumerate(self._engines):
            o = np.asarray(e.reset())
            obs_parts.append(self._pad(o, *self._shapes[gi]))
        self._obs = np.concatenate(obs_parts, axis=0)
        return self._obs, {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        obs_parts, rew_parts, done_parts, trunc_parts = [], [], [], []
        for gi, e in enumerate(self._engines):
            a = actions[gi * self.epg:(gi + 1) * self.epg]
            o, r, d, t, _ = e.step(a)
            obs_parts.append(self._pad(np.asarray(o), *self._shapes[gi]))
            rew_parts.append(np.asarray(r, np.float32))
            done_parts.append(np.asarray(d, bool))
            trunc_parts.append(np.asarray(t, bool))
        self._obs = np.concatenate(obs_parts, axis=0)
        rewards = np.concatenate(rew_parts)
        dones = np.concatenate(done_parts)
        truncs = np.concatenate(trunc_parts)
        infos = {"won": dones, "game_ids": self.game_ids}
        return self._obs, rewards, dones, truncs, infos

    def close(self):
        self._engines = []


if __name__ == "__main__":
    import time
    env = PuzzleScriptVecEnv(["sokoban_basic", "kettle", "Slidings", "Travelling_salesman"],
                             num_envs=4096, max_episode_steps=200)
    obs, _ = env.reset()
    print(f"games={env.games} num_envs={env.num_envs} obs={obs.shape} {obs.dtype} "
          f"act={env.single_action_space.n}")
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(rng.integers(0, 5, env.num_envs))
    N = 200
    t0 = time.time()
    for _ in range(N):
        env.step(rng.integers(0, 5, env.num_envs))
    dt = time.time() - t0
    print(f"throughput: {env.num_envs * N / dt:,.0f} env-steps/sec (4 games padded)")
