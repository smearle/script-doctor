"""Paired final-carry rollouts with and without XLA input-buffer donation."""
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
from scripts.benchmarks.benchmark_chunked_rollouts import assert_equal_on_device
from scripts.benchmarks.benchmark_paper_throughput import make_random_rollout, save_result


def benchmark(game, batch, steps, trials, seeds):
    env = init_ps_env(game, 0, np.iinfo(np.int32).max)
    params = PJParams(level=env.get_level(0), level_i=0)
    key = jax.random.PRNGKey(0)
    _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(jax.random.split(key, batch), params)
    initial = (state, key)
    runners, compilation = [], []
    for donate in (False, True):
        start = time.perf_counter()
        runner = jax.jit(make_random_rollout(env, params, batch, steps), donate_argnums=(0,) if donate else ()).lower(initial).compile()
        memory = runner.memory_analysis()
        compilation.append({"compile_s": time.perf_counter() - start,
                            "alias_bytes": memory.alias_size_in_bytes, "temporary_bytes": memory.temp_size_in_bytes,
                            "argument_bytes": memory.argument_size_in_bytes, "output_bytes": memory.output_size_in_bytes})
        runners.append(runner)
    rows = []
    for seed in seeds:
        carries = [jax.tree.map(lambda x: jnp.array(x, copy=True), (state, jax.random.PRNGKey(seed))) for _ in runners]
        for _ in range(2):
            carries = [jax.block_until_ready(fn(c)) for fn, c in zip(runners, carries)]
            assert_equal_on_device(*carries)
        samples = [[], []]
        for trial in range(trials):
            for i in ((0, 1) if trial % 2 == 0 else (1, 0)):
                start = time.perf_counter()
                carries[i] = jax.block_until_ready(runners[i](carries[i]))
                samples[i].append(time.perf_counter() - start)
            assert_equal_on_device(*carries)
        row = {"game": game, "batch": batch, "steps": steps, "seed": seed,
               "samples_s": samples, "speedup": float(np.median(samples[0]) / np.median(samples[1])),
               "compilation": compilation, "exact_final_carries": True}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[256, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1042])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = {"devices": [d.device_kind for d in jax.devices()], "jax_version": jax.__version__,
              "trials": args.trials, "seeds": args.seeds, "steps": args.steps,
              "timing": "alternating warmed continuing final-carry rollouts; same actions and state; equality outside timer",
              "source_hashes": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                ["puzzlescript_jax/env.py", "scripts/benchmarks/benchmark_paper_throughput.py",
                                 str(Path(__file__).relative_to(root))]}, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        for batch in args.batches:
            print(f"Compiling {game}, batch {batch}", flush=True)
            result["results"].extend(benchmark(game, batch, args.steps, args.trials, args.seeds))
            save_result(args.output, result)
            jax.clear_caches()


if __name__ == "__main__":
    main()
