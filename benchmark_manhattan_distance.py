import statistics
import time

import jax
import jax.numpy as jnp


INT_INF = jnp.int32(1 << 29)


def safe_add_one(x):
    return jnp.where(x >= INT_INF, INT_INF, x + 1)


def pairwise_sum_nearest(src_channel, trg_channel):
    n_cells = src_channel.size
    src_coords = jnp.argwhere(src_channel, size=n_cells, fill_value=-1)
    trg_coords = jnp.argwhere(trg_channel, size=n_cells, fill_value=-1)
    src_coords = src_coords[:, None, :]
    trg_coords = trg_coords[None, :, :]
    dists = jnp.abs(src_coords - trg_coords).sum(axis=-1)
    dists = jnp.where(jnp.all(src_coords == -1, axis=-1), jnp.nan, dists)
    dists = jnp.where(jnp.all(trg_coords == -1, axis=-1), jnp.nan, dists)
    dists = jnp.nanmin(dists, axis=1)
    dists = jnp.where(jnp.isnan(dists), 0, dists)
    return jnp.sum(dists).astype(jnp.int32)


def pairwise_min_nearest(src_channel, trg_channel):
    n_cells = src_channel.size
    src_coords = jnp.argwhere(src_channel, size=n_cells, fill_value=-1)
    trg_coords = jnp.argwhere(trg_channel, size=n_cells, fill_value=-1)
    src_coords = src_coords[:, None, :]
    trg_coords = trg_coords[None, :, :]
    dists = jnp.abs(src_coords - trg_coords).sum(axis=-1)
    dists = jnp.where(jnp.all(src_coords == -1, axis=-1), jnp.nan, dists)
    dists = jnp.where(jnp.all(trg_coords == -1, axis=-1), jnp.nan, dists)
    dists = jnp.where(jnp.isnan(dists), INT_INF, dists)
    return jnp.min(dists).astype(jnp.int32)


def manhattan_distance_transform(trg_channel):
    h, w = trg_channel.shape
    dist0 = jnp.where(trg_channel, 0, INT_INF).astype(jnp.int32)

    def forward_row(y, dist):
        def forward_col(x, dist_inner):
            val = dist_inner[y, x]
            val = jnp.minimum(val, jnp.where(y > 0, safe_add_one(dist_inner[y - 1, x]), INT_INF))
            val = jnp.minimum(val, jnp.where(x > 0, safe_add_one(dist_inner[y, x - 1]), INT_INF))
            return dist_inner.at[y, x].set(val)

        return jax.lax.fori_loop(0, w, forward_col, dist)

    dist = jax.lax.fori_loop(0, h, forward_row, dist0)

    def backward_row(y_rev, dist_inner):
        y = h - 1 - y_rev

        def backward_col(x_rev, dist_last):
            x = w - 1 - x_rev
            val = dist_last[y, x]
            val = jnp.minimum(val, jnp.where(y + 1 < h, safe_add_one(dist_last[y + 1, x]), INT_INF))
            val = jnp.minimum(val, jnp.where(x + 1 < w, safe_add_one(dist_last[y, x + 1]), INT_INF))
            return dist_last.at[y, x].set(val)

        return jax.lax.fori_loop(0, w, backward_col, dist_inner)

    return jax.lax.fori_loop(0, h, backward_row, dist)


def transform_sum_nearest(src_channel, trg_channel):
    dist_map = manhattan_distance_transform(trg_channel)
    return jnp.where(jnp.any(trg_channel), jnp.sum(jnp.where(src_channel, dist_map, 0)), 0).astype(jnp.int32)


def transform_min_nearest(src_channel, trg_channel):
    dist_map = manhattan_distance_transform(trg_channel)
    masked = jnp.where(src_channel, dist_map, INT_INF)
    return jnp.min(masked).astype(jnp.int32)


def make_case(shape, src_density, trg_density, seed):
    key = jax.random.PRNGKey(seed)
    key_src, key_trg = jax.random.split(key)
    src = jax.random.bernoulli(key_src, p=src_density, shape=shape)
    trg = jax.random.bernoulli(key_trg, p=trg_density, shape=shape)
    return src, trg


def make_batch(shape, src_density, trg_density, seed, batch_size):
    srcs = []
    trgs = []
    for i in range(batch_size):
        src, trg = make_case(shape, src_density, trg_density, seed + i)
        srcs.append(src)
        trgs.append(trg)
    return jnp.stack(srcs), jnp.stack(trgs)


def make_single_runner(fn, inner_reps):
    def runner(src, trg):
        init = fn(src, trg)

        def body(_, acc):
            return acc + fn(src, trg)

        return jax.lax.fori_loop(0, inner_reps, body, init)

    return runner


def make_batched_runner(fn, inner_reps):
    batched_fn = jax.vmap(fn, in_axes=(0, 0))

    def runner(src_batch, trg_batch):
        init = batched_fn(src_batch, trg_batch)

        def body(_, acc):
            return acc + batched_fn(src_batch, trg_batch)

        return jax.lax.fori_loop(0, inner_reps, body, init)

    return runner


def compile_runner(runner, *args):
    t0 = time.perf_counter()
    compiled = jax.jit(runner).lower(*args).compile()
    jax.block_until_ready(compiled(*args))
    return compiled, time.perf_counter() - t0


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


def correctness_suite():
    print("correctness")
    cases = [
        ((4, 4), 0.0, 0.0, 0),
        ((4, 4), 0.25, 0.25, 1),
        ((8, 8), 0.10, 0.10, 2),
        ((8, 8), 0.50, 0.50, 3),
        ((16, 16), 0.05, 0.20, 4),
        ((16, 16), 0.30, 0.05, 5),
    ]
    for shape, src_d, trg_d, seed in cases:
        src, trg = make_case(shape, src_d, trg_d, seed)
        pair_sum = int(pairwise_sum_nearest(src, trg))
        xform_sum = int(transform_sum_nearest(src, trg))
        pair_min = int(pairwise_min_nearest(src, trg))
        xform_min = int(transform_min_nearest(src, trg))
        print(
            f"shape={shape} src_d={src_d:.2f} trg_d={trg_d:.2f} "
            f"sum_match={pair_sum == xform_sum} min_match={pair_min == xform_min}"
        )
        if pair_sum != xform_sum or pair_min != xform_min:
            raise SystemExit(
                f"mismatch for shape={shape}: "
                f"pair_sum={pair_sum}, xform_sum={xform_sum}, "
                f"pair_min={pair_min}, xform_min={xform_min}"
            )
    print("all correctness checks passed")


def benchmark_metric(name, fn_a, fn_b, shape, src_d, trg_d, seed, inner_reps, outer_trials, batch_size):
    src, trg = make_case(shape, src_d, trg_d, seed)
    src_batch, trg_batch = make_batch(shape, src_d, trg_d, seed + 1000, batch_size)

    a_single_runner = make_single_runner(fn_a, inner_reps)
    a_single_compiled, a_single_compile_s = compile_runner(a_single_runner, src, trg)
    a_single_stats = benchmark_runner(a_single_compiled, (src, trg), inner_reps + 1, outer_trials)

    b_single_runner = make_single_runner(fn_b, inner_reps)
    b_single_compiled, b_single_compile_s = compile_runner(b_single_runner, src, trg)
    b_single_stats = benchmark_runner(b_single_compiled, (src, trg), inner_reps + 1, outer_trials)

    a_batch_runner = make_batched_runner(fn_a, inner_reps)
    a_batch_compiled, a_batch_compile_s = compile_runner(a_batch_runner, src_batch, trg_batch)
    a_batch_stats = benchmark_runner(
        a_batch_compiled,
        (src_batch, trg_batch),
        (inner_reps + 1) * batch_size,
        outer_trials,
    )

    b_batch_runner = make_batched_runner(fn_b, inner_reps)
    b_batch_compiled, b_batch_compile_s = compile_runner(b_batch_runner, src_batch, trg_batch)
    b_batch_stats = benchmark_runner(
        b_batch_compiled,
        (src_batch, trg_batch),
        (inner_reps + 1) * batch_size,
        outer_trials,
    )

    print(
        f"{shape},{src_d:.2f},{trg_d:.2f},{name},"
        f"{a_single_compile_s * 1e3:.2f},{b_single_compile_s * 1e3:.2f},"
        f"{a_single_stats['median_us']:.2f},{b_single_stats['median_us']:.2f},{a_single_stats['median_us'] / b_single_stats['median_us']:.2f},"
        f"{a_batch_compile_s * 1e3:.2f},{b_batch_compile_s * 1e3:.2f},"
        f"{a_batch_stats['median_us']:.2f},{b_batch_stats['median_us']:.2f},{a_batch_stats['median_us'] / b_batch_stats['median_us']:.2f}"
    )


def speed_suite():
    inner_reps = 200
    outer_trials = 7
    batch_size = 64
    print("\nspeed")
    print(f"inner_reps={inner_reps},outer_trials={outer_trials},batch_size={batch_size}")
    print(
        "shape,src_density,trg_density,metric,"
        "pair_compile_ms,xform_compile_ms,pair_us,xform_us,speedup,"
        "pair_batch_compile_ms,xform_batch_compile_ms,pair_batch_us,xform_batch_us,batch_speedup"
    )
    cases = [
        ((16, 16), 0.10, 0.10, 20),
        ((32, 32), 0.05, 0.05, 21),
        ((32, 32), 0.20, 0.20, 22),
        ((64, 64), 0.05, 0.05, 23),
        ((64, 64), 0.20, 0.20, 24),
    ]
    for shape, src_d, trg_d, seed in cases:
        benchmark_metric(
            "sum",
            pairwise_sum_nearest,
            transform_sum_nearest,
            shape,
            src_d,
            trg_d,
            seed,
            inner_reps,
            outer_trials,
            batch_size,
        )
        benchmark_metric(
            "min",
            pairwise_min_nearest,
            transform_min_nearest,
            shape,
            src_d,
            trg_d,
            seed + 500,
            inner_reps,
            outer_trials,
            batch_size,
        )


if __name__ == "__main__":
    correctness_suite()
    speed_suite()
