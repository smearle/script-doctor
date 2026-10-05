"""Chunk boundaries must not alter keys, environment order, or autoresets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_chunked_rollouts import make_rollout
from scripts.benchmarks.benchmark_manhattan_distance import assert_equal


class StochasticCounter:
    def step(self, key, state, action, params):
        noise = jax.random.randint(key, (), 0, 97)
        time = state["time"] + 1
        done = time == 3
        value = jnp.where(done, noise, state["value"] + noise + action)
        state = {"value": value, "time": jnp.where(done, 0, time)}
        return value, state, noise.astype(jnp.float32), done, {"noise": noise}


@pytest.mark.parametrize("chunk", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["full", "final"])
def test_chunk_boundaries_preserve_all_streams(chunk, mode):
    state = {"value": jnp.arange(8), "time": jnp.arange(8) % 3}
    actions = jnp.asarray(np.random.default_rng(7).integers(0, 5, (11, 8), dtype=np.int32))
    args = (state, jax.random.PRNGKey(21), actions)
    env = StochasticCounter()
    before = jax.jit(make_rollout(env, None, mode=mode))(*args)
    after = jax.jit(make_rollout(env, None, chunk, mode))(*args)
    assert_equal(before, after)


def test_actual_random_rules_and_autoresets():
    env = init_ps_env("test_random", level_i=0, max_episode_steps=2)
    params = PJParams(level=env.get_level(0), level_i=0)
    _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(jax.random.split(jax.random.PRNGKey(0), 4), params)
    args = (state, jax.random.PRNGKey(11), jnp.arange(28).reshape(7, 4) % env.action_space.n)
    before = jax.jit(make_rollout(env, params))(*args)
    after = jax.jit(make_rollout(env, params, 2))(*args)
    assert_equal(before, after)
