import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env


def compile_runner(fn, *args):
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    jax.block_until_ready(compiled(*args))
    return compiled, time.perf_counter() - t0


def summarize_output(tree):
    leaves = jax.tree.leaves(tree)
    total = jnp.int32(0)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.dtype == jnp.bool_:
            arr = arr.astype(jnp.int32)
        total = total + jnp.sum(arr).astype(jnp.int32)
    return total


def benchmark_runner(compiled, args, denom, outer_trials):
    per_item_us = []
    for _ in range(outer_trials):
        t0 = time.perf_counter()
        out = compiled(*args)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        per_item_us.append((dt / denom) * 1e6)
    return {
        "median_us": statistics.median(per_item_us),
        "min_us": min(per_item_us),
        "max_us": max(per_item_us),
    }


def make_single_runner(fn, inner_reps):
    def runner(*args):
        init = summarize_output(fn(*args))

        def body(_, acc):
            return acc + summarize_output(fn(*args))

        return jax.lax.fori_loop(0, inner_reps, body, init)

    return runner


def make_batched_runner(fn, inner_reps, in_axes):
    batched_fn = jax.vmap(fn, in_axes=in_axes)

    def runner(*args):
        init = summarize_output(batched_fn(*args))

        def body(_, acc):
            return acc + summarize_output(batched_fn(*args))

        return jax.lax.fori_loop(0, inner_reps, body, init)

    return runner


def benchmark_impl(name, fn, single_args, batch_args, inner_reps, outer_trials, batch_size, in_axes):
    single_runner = make_single_runner(fn, inner_reps)
    single_compiled, single_compile_s = compile_runner(single_runner, *single_args)
    single_stats = benchmark_runner(single_compiled, single_args, inner_reps + 1, outer_trials)

    batch_runner = make_batched_runner(fn, inner_reps, in_axes=in_axes)
    batch_compiled, batch_compile_s = compile_runner(batch_runner, *batch_args)
    batch_stats = benchmark_runner(
        batch_compiled,
        batch_args,
        (inner_reps + 1) * batch_size,
        outer_trials,
    )

    return (
        name,
        single_compile_s,
        single_stats,
        batch_compile_s,
        batch_stats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="sokoban_basic")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inner-reps", type=int, default=1000)
    parser.add_argument("--outer-trials", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--vmap", action="store_true", default=True)
    args = parser.parse_args()

    env = init_ps_env(args.game, args.level, args.max_steps, vmap=args.vmap)
    level = env.get_level(args.level if args.level >= 0 else 0)
    params = PJParams(level=level, level_i=args.level)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, params)
    del obs

    batch_size = args.batch_size
    reset_keys = jax.random.split(jax.random.PRNGKey(1), batch_size)
    _, states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)

    action = jnp.int32(0)
    actions = jnp.zeros((batch_size,), dtype=jnp.int32)

    force_lvl = env.apply_player_force(action, state)
    batch_force_lvl = jax.vmap(env.apply_player_force, in_axes=(0, 0))(actions, states)

    step_rng = jax.random.PRNGKey(2)
    step_rngs = jax.random.split(step_rng, batch_size)

    cases = [
        (
            "check_win",
            lambda lvl: env.check_win(lvl),
            (state.multihot_level,),
            (states.multihot_level,),
            (0,),
        ),
        (
            "apply_player_force",
            lambda action, state: env.apply_player_force(action, state),
            (action, state),
            (actions, states),
            (0, 0),
        ),
        (
            "tick_fn",
            lambda rng, lvl: env.tick_fn(rng, lvl),
            (step_rng, force_lvl),
            (step_rngs, batch_force_lvl),
            (0, 0),
        ),
        (
            "step_env",
            lambda rng, state, action, params: env.step_env(rng, state, action, params),
            (step_rng, state, action, params),
            (step_rngs, states, actions, params),
            (0, 0, 0, None),
        ),
    ]

    print(
        f"game={args.game},level={args.level},batch_size={batch_size},"
        f"inner_reps={args.inner_reps},outer_trials={args.outer_trials}"
    )
    print(
        "name,"
        "single_compile_ms,single_us,single_min_us,single_max_us,"
        "batch_compile_ms,batch_us,batch_min_us,batch_max_us"
    )

    for name, fn, single_args, batch_args, in_axes in cases:
        row = benchmark_impl(
            name,
            fn,
            single_args,
            batch_args,
            args.inner_reps,
            args.outer_trials,
            batch_size,
            in_axes,
        )
        _, single_compile_s, single_stats, batch_compile_s, batch_stats = row
        print(
            f"{name},"
            f"{single_compile_s * 1e3:.3f},{single_stats['median_us']:.3f},{single_stats['min_us']:.3f},{single_stats['max_us']:.3f},"
            f"{batch_compile_s * 1e3:.3f},{batch_stats['median_us']:.3f},{batch_stats['min_us']:.3f},{batch_stats['max_us']:.3f}"
        )


if __name__ == "__main__":
    main()
