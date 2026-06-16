import statistics
import time

import jax
import jax.numpy as jnp


def make_funcs(n_funcs, n_channels):
    funcs = []
    for i in range(n_funcs):
        src = i % n_channels
        trg = (i + 1) % n_channels
        mode = i % 4

        if mode == 0:
            def fn(lvl, src=src, trg=trg):
                src_ch = lvl[src]
                trg_ch = lvl[trg]
                overlap = src_ch & trg_ch
                win = jnp.any(overlap)
                score = jnp.count_nonzero(overlap).astype(jnp.int32)
                heuristic = -score
                return win, score, heuristic
        elif mode == 1:
            def fn(lvl, src=src):
                src_ch = lvl[src]
                count = jnp.count_nonzero(src_ch).astype(jnp.int32)
                win = ~jnp.any(src_ch)
                score = -count
                return win, score, score
        elif mode == 2:
            def fn(lvl, src=src, trg=trg):
                src_ch = lvl[src]
                trg_ch = lvl[trg]
                win = ~jnp.any(src_ch & ~trg_ch)
                score = jnp.count_nonzero(src_ch & trg_ch).astype(jnp.int32)
                heuristic = -jnp.count_nonzero(src_ch ^ trg_ch).astype(jnp.int32)
                return win, score, heuristic
        else:
            def fn(lvl, src=src):
                src_ch = lvl[src]
                count = jnp.count_nonzero(src_ch).astype(jnp.int32)
                win = count > 0
                return win, count, -count

        funcs.append(fn)
    return tuple(funcs)


def check_win_vmap_switch(lvl, funcs):
    def apply_win_condition_func(i, lvl):
        return jax.lax.switch(i, funcs, lvl)

    wins, scores, heuristics = jax.vmap(apply_win_condition_func, in_axes=(0, None))(jnp.arange(len(funcs)), lvl)
    return jnp.all(wins), scores.sum(), heuristics.sum()


def check_win_stack(lvl, funcs):
    outs = [f(lvl) for f in funcs]
    wins = jnp.stack([o[0] for o in outs])
    scores = jnp.stack([o[1] for o in outs])
    heuristics = jnp.stack([o[2] for o in outs])
    return jnp.all(wins), scores.sum(), heuristics.sum()


def check_win_python_loop(lvl, funcs):
    all_win = True
    total_score = jnp.int32(0)
    total_heuristic = jnp.int32(0)
    for f in funcs:
        win, score, heuristic = f(lvl)
        all_win = all_win & win
        total_score = total_score + score
        total_heuristic = total_heuristic + heuristic
    return all_win, total_score, total_heuristic


def check_win_lax_map(lvl, funcs):
    def body(i):
        return jax.lax.switch(i, funcs, lvl)

    wins, scores, heuristics = jax.lax.map(body, jnp.arange(len(funcs)))
    return jnp.all(wins), scores.sum(), heuristics.sum()


def make_repeated_runner(check_fn, funcs, inner_reps):
    def run(lvl):
        def body(_, acc):
            win, score, heuristic = check_fn(lvl, funcs)
            return (
                acc[0] ^ win,
                acc[1] + score,
                acc[2] + heuristic,
            )

        init = (jnp.array(False), jnp.int32(0), jnp.int32(0))
        return jax.lax.fori_loop(0, inner_reps, body, init)

    return run


def compile_runner(check_fn, lvl, funcs, inner_reps):
    runner = make_repeated_runner(check_fn, funcs, inner_reps)
    t0 = time.perf_counter()
    compiled = jax.jit(runner).lower(lvl).compile()
    jax.block_until_ready(compiled(lvl))
    compile_s = time.perf_counter() - t0
    return compiled, compile_s


def benchmark_runner(compiled, lvl, inner_reps, outer_trials):
    per_call_us = []
    for _ in range(outer_trials):
        t0 = time.perf_counter()
        out = compiled(lvl)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        per_call_us.append((dt / inner_reps) * 1e6)
    return {
        "median_us": statistics.median(per_call_us),
        "min_us": min(per_call_us),
        "max_us": max(per_call_us),
    }


def verify_equivalence(lvl, funcs):
    ref = check_win_stack(lvl, funcs)
    for name, fn in [
        ("vmap_switch", check_win_vmap_switch),
        ("stack", check_win_stack),
        ("python_loop", check_win_python_loop),
        ("lax_map", check_win_lax_map),
    ]:
        out = fn(lvl, funcs)
        if not all(bool(jnp.array_equal(a, b)) for a, b in zip(ref, out)):
            raise RuntimeError(f"{name} mismatch")


def run_case(n_funcs, shape, density, seed, inner_reps, outer_trials):
    n_channels = max(8, n_funcs)
    key = jax.random.PRNGKey(seed)
    lvl = jax.random.bernoulli(key, p=density, shape=(n_channels, *shape))
    funcs = make_funcs(n_funcs, n_channels)
    verify_equivalence(lvl, funcs)

    rows = []
    for name, fn in [
        ("vmap_switch", check_win_vmap_switch),
        ("stack", check_win_stack),
        ("python_loop", check_win_python_loop),
        ("lax_map", check_win_lax_map),
    ]:
        compiled, compile_s = compile_runner(fn, lvl, funcs, inner_reps)
        stats = benchmark_runner(compiled, lvl, inner_reps, outer_trials)
        rows.append((name, compile_s, stats))
    return rows


if __name__ == "__main__":
    inner_reps = 1000
    outer_trials = 7
    print(f"inner_reps={inner_reps},outer_trials={outer_trials}")
    print("n_funcs,shape,impl,compile_ms,median_us,min_us,max_us")
    cases = [
        (1, (8, 8), 0.2, 0),
        (4, (8, 8), 0.2, 1),
        (8, (8, 8), 0.2, 2),
        (16, (8, 8), 0.2, 3),
        (4, (32, 32), 0.2, 4),
        (8, (32, 32), 0.2, 5),
        (16, (32, 32), 0.2, 6),
    ]
    for n_funcs, shape, density, seed in cases:
        for name, compile_s, stats in run_case(
            n_funcs, shape, density, seed, inner_reps=inner_reps, outer_trials=outer_trials
        ):
            print(
                f"{n_funcs},{shape},{name},{compile_s * 1e3:.3f},"
                f"{stats['median_us']:.3f},{stats['min_us']:.3f},{stats['max_us']:.3f}"
            )
