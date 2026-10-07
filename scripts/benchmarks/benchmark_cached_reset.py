"""Test fixed-level deterministic reset specialization without changing the engine.

This experiment specializes a rollout to one immutable level/parameter value.
It deliberately refuses random games and multi-level environments. It is not a
replacement for the general reset API, whose level parameter may change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env
from scripts.benchmarks.benchmark_chunked_rollouts import make_rollout, assert_equal_on_device
from scripts.benchmarks.benchmark_paper_throughput import save_result


def specialize_reset(env, params):
    if env.has_randomness() or env._is_multi_level:
        raise ValueError('Only deterministic single-level environments can use this experiment')
    key = jax.random.PRNGKey(0)
    obs, state = jax.block_until_ready(env.reset(key, params))
    assert_equal_on_device(state.rng, key)
    for seed in (42, 1042):
        other_key = jax.random.PRNGKey(seed)
        other_obs, other_state = jax.block_until_ready(env.reset(other_key, params))
        assert_equal_on_device((obs, state.replace(rng=other_key)), (other_obs, other_state))

    def reset(rng, _supplied_params):
        # This experimental environment is used only by make_rollout below,
        # which closes over the fixed params used to compute this template.
        return obs, state.replace(rng=rng)

    env.reset = reset


def benchmark(game, batch, steps, trials, seeds, safe_api=False):
    runners, stats, initial_states = [], [], []
    reset_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    actions = jnp.asarray(np.random.default_rng(42).integers(0, 5, (steps, batch), dtype=np.int32))
    for cached in (False, True):
        env = init_ps_env(game, 0, max_episode_steps=17)
        params = PJParams(level=env.get_level(0), level_i=0)
        if cached:
            if safe_api:
                params = env.prepare_params(params)
            else:
                specialize_reset(env, params)
            _, state = jax.vmap(lambda k: env.reset(k, params))(reset_keys)
        else:
            _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(reset_keys, params)
        initial_states.append(state)
        start = time.perf_counter()
        runner = jax.jit(make_rollout(env, params, mode='full')).lower(state, jax.random.PRNGKey(1), actions).compile()
        memory = runner.memory_analysis()
        stats.append({'compile_s': time.perf_counter() - start, 'temporary_bytes': memory.temp_size_in_bytes,
                      'output_bytes': memory.output_size_in_bytes})
        runners.append(runner)
    assert_equal_on_device(*initial_states)
    rows = []
    for seed in seeds:
        actions = jnp.asarray(np.random.default_rng(seed).integers(0, 5, (steps, batch), dtype=np.int32))
        args = (initial_states[0], jax.random.PRNGKey(seed), actions)
        outputs = [jax.block_until_ready(fn(*args)) for fn in runners]
        assert_equal_on_device(*outputs)
        done_count = int(jnp.count_nonzero(outputs[0][1][3]))
        del outputs
        for _ in range(2):
            for fn in runners:
                jax.block_until_ready(fn(*args))
        samples = [[], []]
        for trial in range(trials):
            for i in ((0, 1) if trial % 2 == 0 else (1, 0)):
                start = time.perf_counter()
                jax.block_until_ready(runners[i](*args))
                samples[i].append(time.perf_counter() - start)
        row = {'game': game, 'batch': batch, 'steps': steps, 'seed': seed, 'samples_s': samples,
               'speedup': float(np.median(samples[0]) / np.median(samples[1])), 'compilation': stats,
               'exact_full_outputs': True, 'automatic_resets': done_count}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', nargs='+', required=True)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1042])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--safe-api', action='store_true', help='Use prepare_params, including cache invalidation, rather than the experiment-only specialization.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = {'devices': [d.device_kind for d in jax.devices()], 'jax_version': jax.__version__,
              'trials': args.trials, 'safe_api': args.safe_api, 'max_episode_steps': 17, 'scope': 'immutable fixed level, deterministic games only',
              'timing': 'alternating warmed full-output calls; same initial states and actions; exact comparisons outside timer',
              'source_hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                ['puzzlescript_jax/env.py', str(Path(__file__).relative_to(root))]}, 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        result['results'].extend(benchmark(game, args.batch, args.steps, args.trials, args.seeds, args.safe_api))
        save_result(args.output, result)
        jax.clear_caches()


if __name__ == '__main__':
    main()
