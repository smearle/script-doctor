# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A demonstration of the policy improvement by planning with Gumbel."""

import time
import functools
import traceback
from typing import Tuple
import hydra
import itertools
import chex
import jax
import jax.numpy as jnp
import env as Env
import mctx
from puzzlejax.utils import get_list_of_games_for_testing
from utils_rl import get_env_params_from_config, init_ps_env
from conf.config import PSConfig
import os,logging

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 1=INFO off, 2=INFO+WARNING off, 3=INFO+WARN+ERR off

# Quiet Python loggers
logging.basicConfig(level=logging.WARNING)          # set root to WARNING
logging.getLogger("jax").setLevel(logging.ERROR)    # or WARNING
logging.getLogger("jax._src").setLevel(logging.ERROR)

SEED = 42
MAX_ACTION = 100
BATCH_SIZES = [ # number of games to play with mctx
    2,
    5,
]
# batch_sizes = batch_sizes[::-1]
VMAPS = [
    True,
    # False,
]

@chex.dataclass(frozen=True)
class DemoOutput:
  prior_policy_value: chex.Array
  prior_policy_action_value: chex.Array
  selected_action_value: chex.Array
  action_weights_policy_value: chex.Array


def _run_demo(rng_key: chex.PRNGKey, cfg : PSConfig) -> Tuple[chex.PRNGKey, DemoOutput]:
  """Runs a search algorithm on random data."""
  rng_key, logits_rng, step_rng, search_rng = jax.random.split(rng_key, 4)
  
  use_mixed_value = False
  games = get_list_of_games_for_testing(all_games=False)

  global BATCH_SIZES, VMAPS
  hparams = itertools.product(games,
      BATCH_SIZES,
      VMAPS,
  )
  games, BATCH_SIZES, VMAPS = zip(*hparams)

  last_game = None
  for (game, n_envs, vmap) in zip(games, BATCH_SIZES, VMAPS):
    try:
      cfg.game = game
      cfg.vmap = vmap

      jax.debug.print(f'\nGame: {game}, n_envs: {n_envs}, vmap: {vmap}.')
      raw_value = jax.random.normal(logits_rng, shape=[n_envs])
      init_ps_env(cfg)

      if last_game != game:
          env = init_ps_env(cfg)

      last_game = game

      env_params = get_env_params_from_config(env, cfg)

      # INIT ENV
      rng, _rng = jax.random.split(rng_key)
      reset_rng = jax.random.split(_rng, n_envs)
      level = env.get_level(0)
      params = Env.PSParams(level=level)
      key = jax.random.PRNGKey(0)
      obsv, init_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
      mc_envs = 80000
      mc_steps = 30

      def monte_carlo_step(carry, unused):
        env_state, rng, _ = carry
        rng, _rng = jax.random.split(rng)
        rand_act = jax.random.randint(_rng, (mc_envs,), 0, env.action_space.n)
        # STEP ENV
        rng_step = jax.random.split(_rng, mc_envs)
        _, env_state, _, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, rand_act, env_params)
        carry = (env_state, rng, done)
        return carry, None
      
      def single_rollout(carry0):
        return jax.lax.scan(monte_carlo_step, carry0, None, length=mc_steps)
      
      _rollout_jitted = jax.jit(single_rollout)

      def _make_bandit_recurrent_fn():
        """Returns a recurrent_fn for a determistic bandit."""
        def recurrent_fn(params, rng_key, action, env_state):
          del params

          rand_act = jax.random.randint(rng_key, (n_envs, env.action_space.n), 0, env.action_space.n).astype(jnp.float32)
          rng_step = jax.random.split(rng_key, n_envs)
          obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)

          rng_rollout = jax.random.split(rng_key, n_envs)
          carry = (env_state, rng_rollout, jax.numpy.zeros((n_envs,), dtype=jax.numpy.bool_))
          #carry, _ = jax.lax.scan(_monte_carlo_jitted, carry, None, mc_step)
          #carry0 = jnp.tile(carry, (n_envs, mc_envs))
          (carryT, rngT, doneT), rewards = jax.vmap(_rollout_jitted)(carry)
          #jax.debug.print("rollout wins: {}", doneT)
          jax.debug.print("env_states: {}", env_state)
          #wins = jax.numpy.sum(doneT)

          # On a single-player environment, use discount from [0, 1].
          # On a zero-sum self-play environment, use discount=-1.
          discount = jnp.ones_like(reward)
          #jax.debug.print("reward = {}", reward)
          recurrent_fn_output = mctx.RecurrentFnOutput(
              reward=reward,
              discount=discount,
              prior_logits=rand_act,
              value=jnp.zeros_like(reward))
          return recurrent_fn_output, env_state

        return recurrent_fn



      _env_step_jitted = jax.jit(monte_carlo_step)


      start = time.time()
      env_state = init_state
      done = False
      step = 0
      while not done:
        step+=1
        rand_act = jax.random.randint(_rng, (n_envs, env.action_space.n), 0, env.action_space.n).astype(jnp.float32)
        # The root output would be the output of MuZero representation network.
        root = mctx.RootFnOutput(
            prior_logits=rand_act,
            value=raw_value,
            # The embedding is used only to implement the MuZero model.
            embedding=env_state,
        )
        # The recurrent_fn would be provided by MuZero dynamics network.
        recurrent_fn = _make_bandit_recurrent_fn()

        batch_rng = jax.random.split(step_rng, n_envs)
        # Running the search.
        policy_output = mctx.gumbel_muzero_policy(
            params=(),
            rng_key=search_rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=n_envs,
            max_num_considered_actions=MAX_ACTION,
            qtransform=functools.partial(
                mctx.qtransform_completed_by_mix_value,
                use_mixed_value=use_mixed_value),)
        
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(batch_rng, env_state, policy_output.action, env_params)
        jax.debug.print("Step {} Avg reward: {}", step, jnp.mean(reward))
        done = bool(jnp.any(done))
        if step == 5:
          break

      took = time.time() - start
      print(f"Execution Time of {step} steps: {took} seconds")
      print(f"Per Step Per Game Time: {took / n_envs / step}")
      
    except Exception as e:
        # 🔹 Log failure
        with open("mctx_profile_list.txt", "a") as f:
            f.write(f"{game, n_envs, vmap} FAILED: {e}\n")
        # (Optional: also dump stacktrace to console)
        traceback.print_exc()

    else:
        # 🔹 Log success
        with open("mctx_profile_list.txt", "a") as f:
            f.write(f"{game, n_envs, vmap} SUCCESS\n")


  output = DemoOutput(
      prior_policy_value=raw_value,
      prior_policy_action_value=0,
      selected_action_value=0,
      action_weights_policy_value=0,
  )
  return rng_key, output



@hydra.main(version_base="1.3", config_path='./conf', config_name='profile_jax')
def main(cfg : PSConfig):
  rng_key = jax.random.PRNGKey(42)
  _run_demo(rng_key,cfg)

if __name__ == "__main__":
  main()
