import statistics
import time

import jax
import jax.numpy as jnp


def argwhere_first_1d(mask):
    return jnp.argwhere(mask, size=1, fill_value=-1)[0, 0]


def argmax_first_1d(mask):
    idx = jnp.argmax(mask)
    return jnp.where(jnp.any(mask), idx, -1)


def argwhere_first_2d(mask):
    return jnp.argwhere(mask, size=1, fill_value=-1)[0]


def argmax_first_2d(mask):
    flat = mask.reshape(-1)
    flat_idx = jnp.argmax(flat)
    has_any = jnp.any(flat)
    row = flat_idx // mask.shape[1]
    col = flat_idx % mask.shape[1]
    return jnp.where(has_any, jnp.array([row, col], dtype=jnp.int32), jnp.array([-1, -1], dtype=jnp.int32))


def make_mask_1d(size, hit):
    mask = jnp.zeros((size,), dtype=bool)
    if hit == "first":
        mask = mask.at[0].set(True)
    elif hit == "middle":
        mask = mask.at[size // 2].set(True)
    elif hit == "last":
        mask = mask.at[size - 1].set(True)
    return mask


def make_mask_2d(shape, hit):
    rows, cols = shape
    mask = jnp.zeros(shape, dtype=bool)
    if hit == "first":
        mask = mask.at[0, 0].set(True)
    elif hit == "middle":
        mask = mask.at[rows // 2, cols // 2].set(True)
    elif hit == "last":
        mask = mask.at[rows - 1, cols - 1].set(True)
    return mask


def make_batched_masks_1d(size, hit, batch_size):
    hit_cycle = {
        "none": ("none", "none", "none", "none"),
        "first": ("first", "middle", "last", "first"),
        "middle": ("middle", "last", "first", "middle"),
        "last": ("last", "first", "middle", "last"),
    }[hit]
    masks = [make_mask_1d(size, hit_cycle[i % len(hit_cycle)]) for i in range(batch_size)]
    return jnp.stack(masks)


def make_batched_masks_2d(shape, hit, batch_size):
    hit_cycle = {
        "none": ("none", "none", "none", "none"),
        "first": ("first", "middle", "last", "first"),
        "middle": ("middle", "last", "first", "middle"),
        "last": ("last", "first", "middle", "last"),
    }[hit]
    masks = [make_mask_2d(shape, hit_cycle[i % len(hit_cycle)]) for i in range(batch_size)]
    return jnp.stack(masks)


def make_single_runner(fn, inner_reps):
    def runner(sample_input):
        init = fn(sample_input)

        def body(_, acc):
            out = fn(sample_input)
            return jax.tree.map(lambda a, b: a + b, acc, out)

        return jax.lax.fori_loop(0, inner_reps, body, init)

    return runner


def make_batched_runner(fn, inner_reps):
    batched_fn = jax.vmap(fn)

    def runner(batch_input):
        init = batched_fn(batch_input)

        def body(_, acc):
            out = batched_fn(batch_input)
            return jax.tree.map(lambda a, b: a + b, acc, out)

        return jax.lax.fori_loop(0, inner_reps, body, init)

    return runner


def compile_runner(runner, sample_input):
    t0 = time.perf_counter()
    compiled = jax.jit(runner).lower(sample_input).compile()
    jax.block_until_ready(compiled(sample_input))
    return compiled, time.perf_counter() - t0


def benchmark_runner(compiled, sample_input, denom, outer_trials):
    per_item_us = []
    for _ in range(outer_trials):
        t0 = time.perf_counter()
        out = compiled(sample_input)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        per_item_us.append((dt / denom) * 1e6)
    return {
        "median_us": statistics.median(per_item_us),
        "min_us": min(per_item_us),
        "max_us": max(per_item_us),
    }


def measure_case(fn, sample_input, batch_input, inner_reps, outer_trials):
    single_runner = make_single_runner(fn, inner_reps)
    single_compiled, single_compile_s = compile_runner(single_runner, sample_input)
    single_stats = benchmark_runner(single_compiled, sample_input, inner_reps + 1, outer_trials)

    batched_runner = make_batched_runner(fn, inner_reps)
    batched_compiled, batched_compile_s = compile_runner(batched_runner, batch_input)
    batch_size = batch_input.shape[0]
    batched_stats = benchmark_runner(
        batched_compiled,
        batch_input,
        (inner_reps + 1) * batch_size,
        outer_trials,
    )
    return single_compile_s, single_stats, batched_compile_s, batched_stats


def benchmark_1d():
    inner_reps = 1000
    outer_trials = 7
    batch_size = 256
    print("1D benchmark")
    print(f"inner_reps={inner_reps},outer_trials={outer_trials},batch_size={batch_size}")
    print(
        "size,hits,"
        "argwhere_compile_ms,argmax_compile_ms,argwhere_us,argmax_us,speedup,"
        "argwhere_batch_compile_ms,argmax_batch_compile_ms,argwhere_batch_us,argmax_batch_us,batch_speedup"
    )
    for size in (32, 256, 4096, 65536, 1048576):
        for hit in ("none", "first", "middle", "last"):
            sample_input = make_mask_1d(size, hit)
            batch_input = make_batched_masks_1d(size, hit, batch_size)

            aw = measure_case(argwhere_first_1d, sample_input, batch_input, inner_reps, outer_trials)
            am = measure_case(argmax_first_1d, sample_input, batch_input, inner_reps, outer_trials)

            print(
                f"{size},{hit},"
                f"{aw[0] * 1e3:.3f},{am[0] * 1e3:.3f},{aw[1]['median_us']:.3f},{am[1]['median_us']:.3f},{aw[1]['median_us'] / am[1]['median_us']:.3f},"
                f"{aw[2] * 1e3:.3f},{am[2] * 1e3:.3f},{aw[3]['median_us']:.3f},{am[3]['median_us']:.3f},{aw[3]['median_us'] / am[3]['median_us']:.3f}"
            )


def benchmark_2d():
    inner_reps = 1000
    outer_trials = 7
    batch_size = 256
    print("\n2D benchmark")
    print(f"inner_reps={inner_reps},outer_trials={outer_trials},batch_size={batch_size}")
    print(
        "shape,hits,"
        "argwhere_compile_ms,argmax_compile_ms,argwhere_us,argmax_us,speedup,"
        "argwhere_batch_compile_ms,argmax_batch_compile_ms,argwhere_batch_us,argmax_batch_us,batch_speedup"
    )
    for shape in ((8, 8), (32, 32), (128, 128), (512, 512)):
        for hit in ("none", "first", "middle", "last"):
            sample_input = make_mask_2d(shape, hit)
            batch_input = make_batched_masks_2d(shape, hit, batch_size)

            aw = measure_case(argwhere_first_2d, sample_input, batch_input, inner_reps, outer_trials)
            am = measure_case(argmax_first_2d, sample_input, batch_input, inner_reps, outer_trials)

            print(
                f"{shape},{hit},"
                f"{aw[0] * 1e3:.3f},{am[0] * 1e3:.3f},{aw[1]['median_us']:.3f},{am[1]['median_us']:.3f},{aw[1]['median_us'] / am[1]['median_us']:.3f},"
                f"{aw[2] * 1e3:.3f},{am[2] * 1e3:.3f},{aw[3]['median_us']:.3f},{am[3]['median_us']:.3f},{aw[3]['median_us'] / am[3]['median_us']:.3f}"
            )


if __name__ == "__main__":
    benchmark_1d()
    benchmark_2d()
