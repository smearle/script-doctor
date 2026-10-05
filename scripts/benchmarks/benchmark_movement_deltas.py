"""Isolate the constant-table lookup in the retained movement implementation.

Generate the candidate from the exact loaded method, replacing only destination
arithmetic. The assertion intentionally rejects an unfamiliar source version.
This keeps other movement fixes identical in both arms of the experiment.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import jax

import puzzlescript_jax.env as engine
from scripts.benchmarks import benchmark_movement_updates as harness


def arithmetic_movement():
    source = textwrap.dedent(inspect.getsource(engine.PuzzleJaxEnv.apply_movement))
    before = "        delta = deltas[channel % N_MOVEMENTS]\n        x1, y1 = x + delta[0], y + delta[1]"
    after = """        direction = channel % N_MOVEMENTS
        x1 = x + jnp.where(direction == 1, 1, jnp.where(direction == 3, -1, 0))
        y1 = y + jnp.where(direction == 2, 1, jnp.where(direction == 0, -1, 0))"""
    assert source.count(before) == 1, "Movement source changed; review the isolated replacement"
    source = source.replace(before, after)
    namespace = {}
    exec(compile(source, "<movement-delta-candidate>", "exec"), vars(engine), namespace)
    return namespace["apply_movement"], source


def bounded_movement():
    """Move the invariant batched termination reduction outside the loop."""
    source = textwrap.dedent(inspect.getsource(engine.PuzzleJaxEnv.apply_movement))
    before = """            updated, applied, _ = jax.lax.while_loop(
                lambda carry: jnp.any(coordinates[:, carry[2], 0] != -1), body,"""
    after = """            limit = jnp.max(jnp.sum(coordinates[:, :, 0] != -1, axis=1))
            updated, applied, _ = jax.lax.while_loop(
                lambda carry: carry[2] < limit, body,"""
    assert source.count(before) == 1, "Movement source changed; review the isolated replacement"
    source = source.replace(before, after)
    namespace = {}
    exec(compile(source, "<movement-bound-candidate>", "exec"), vars(engine), namespace)
    return namespace["apply_movement"], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["sokoban_basic", "blocks", "Zen_Puzzle_Garden"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 256, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1042])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=("arithmetic", "bounded"), default="arithmetic")
    args = parser.parse_args()
    if min([*args.batches, args.steps, args.trials]) < 1 or min(args.seeds) < 0:
        parser.error("positive sizes and nonnegative seeds required")
    candidate, source = (arithmetic_movement if args.candidate == "arithmetic" else bounded_movement)()
    original = harness.uncached_movement
    harness.uncached_movement = engine.PuzzleJaxEnv.apply_movement
    harness.CANDIDATES[args.candidate] = candidate
    result = {"jax_version": jax.__version__, "devices": [d.device_kind for d in jax.devices()],
              "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "baseline": "current production movement including shared loop and sparse writes",
              "trials": args.trials, "seeds": args.seeds, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for game in args.games:
            for batch in args.batches:
                for row in harness.benchmark(args.candidate, game, 0, batch, args.steps, args.trials, args.seeds):
                    result["results"].append(row)
                    temp = args.output.with_suffix(".json.tmp")
                    temp.write_text(json.dumps(result, indent=2) + "\n")
                    temp.replace(args.output)
                    print(json.dumps(row), flush=True)
                jax.clear_caches()
    finally:
        harness.uncached_movement = original
        harness.CANDIDATES.pop(args.candidate)


if __name__ == "__main__":
    main()
