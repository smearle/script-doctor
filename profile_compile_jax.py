"""Profile JAX compilation time per game.

Default behavior:
- Load games from `data/games_n_rules.json`
- Sort by ascending number of rules
- Measure compile time for a single jitted env step per game
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Iterable

import jax
import jaxlib
import numpy as np

from utils_rl import get_env_params_from_config
from puzzlescript_jax.utils import init_ps_env


@dataclass
class RunConfig:
    game: str
    level: int
    max_episode_steps: int
    vmap: bool


def _normalize_game_name(name: str) -> str:
    return name[:-4] if name.endswith(".txt") else name


def _load_games_n_rules(path: str) -> list[tuple[str, int, bool]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    parsed: list[tuple[str, int, bool]] = []
    for row in data:
        game_name = _normalize_game_name(row[0])
        n_rules = int(row[1])
        has_randomness = bool(row[2]) if len(row) > 2 else False
        parsed.append((game_name, n_rules, has_randomness))
    parsed.sort(key=lambda x: x[1])
    return parsed


def _block_until_ready_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _profile_one_game(game: str, n_rules: int, has_randomness: bool, cfg: RunConfig, n_envs: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "game": game,
        "n_rules": n_rules,
        "has_randomness": has_randomness,
        "n_envs": n_envs,
        "status": "ok",
    }

    game_start = timer()
    try:
        init_start = timer()
        env = init_ps_env(
            game=cfg.game,
            level_i=cfg.level,
            max_episode_steps=cfg.max_episode_steps,
            vmap=cfg.vmap,
        )
        result["env_init_seconds"] = timer() - init_start

        env_params = get_env_params_from_config(env, cfg)
        rng = jax.random.PRNGKey(0)
        rng, reset_key = jax.random.split(rng)
        reset_rng = jax.random.split(reset_key, n_envs)

        reset_start = timer()
        _, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        _block_until_ready_tree(env_state)
        result["reset_seconds"] = timer() - reset_start

        def _env_step(carry):
            state, in_rng = carry
            in_rng, action_key, step_key = jax.random.split(in_rng, 3)
            action = jax.random.randint(action_key, (n_envs,), 0, env.action_space.n)
            step_rng = jax.random.split(step_key, n_envs)
            _, next_state, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_rng, state, action, env_params
            )
            return (next_state, in_rng)

        jit_step = jax.jit(_env_step)
        carry = (env_state, rng)

        compile_start = timer()
        compiled_step = jit_step.lower(carry).compile()
        result["compile_seconds"] = timer() - compile_start

        first_exec_start = timer()
        carry = compiled_step(carry)
        _block_until_ready_tree(carry)
        result["first_execute_seconds"] = timer() - first_exec_start

        second_exec_start = timer()
        carry = compiled_step(carry)
        _block_until_ready_tree(carry)
        result["second_execute_seconds"] = timer() - second_exec_start
        result["total_seconds"] = timer() - game_start

        if n_rules > 0:
            result["compile_seconds_per_rule"] = result["compile_seconds"] / n_rules
        else:
            result["compile_seconds_per_rule"] = None

    except (jaxlib.xla_extension.XlaRuntimeError, Exception) as err:
        result["status"] = "error"
        result["error"] = str(err)
        result["traceback"] = traceback.format_exc()
        result["total_seconds"] = timer() - game_start

    return result


def _summarize(results: Iterable[dict[str, Any]], top_n: int) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok" and "compile_seconds" in r]
    errors = [r for r in results if r.get("status") != "ok"]
    ok_sorted = sorted(ok, key=lambda x: x["compile_seconds"], reverse=True)

    compile_times = np.array([r["compile_seconds"] for r in ok], dtype=float) if ok else np.array([])
    summary: dict[str, Any] = {
        "n_games": len(list(results)) if not isinstance(results, list) else len(results),
        "n_success": len(ok),
        "n_errors": len(errors),
        "slowest_compile_games": ok_sorted[:top_n],
    }
    if compile_times.size > 0:
        summary["compile_seconds_mean"] = float(np.mean(compile_times))
        summary["compile_seconds_median"] = float(np.median(compile_times))
        summary["compile_seconds_p90"] = float(np.percentile(compile_times, 90))
        summary["compile_seconds_max"] = float(np.max(compile_times))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile compile time of PuzzleJax env step per game.")
    parser.add_argument(
        "--games-path",
        type=str,
        default=os.path.join("data", "games_n_rules.json"),
        help="Path to JSON list of [game_name, n_rules, has_randomness].",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join("data", "jax_compile_profile.json"),
        help="Path to write compile-time profiling results JSON.",
    )
    parser.add_argument("--level", type=int, default=0, help="Level index to profile.")
    parser.add_argument("--n-envs", type=int, default=1, help="Batch size for vmapped env step.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=np.iinfo(np.int32).max,
        help="Max episode steps passed to env init.",
    )
    parser.add_argument("--max-games", type=int, default=-1, help="Optional cap for quick tests.")
    parser.add_argument("--skip-randomness", action="store_true", help="Skip games marked as random.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of slowest games to print/save in summary.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_path}. Use --overwrite to replace it."
        )

    games_n_rules = _load_games_n_rules(args.games_path)
    if args.skip_randomness:
        games_n_rules = [row for row in games_n_rules if not row[2]]
    if args.max_games > 0:
        games_n_rules = games_n_rules[: args.max_games]

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one JAX device, found {len(devices)}: {devices}")

    print(
        f"Profiling {len(games_n_rules)} games on {devices[0].device_kind} "
        f"with n_envs={args.n_envs}, ordered by n_rules."
    )

    results: list[dict[str, Any]] = []
    for i, (game, n_rules, has_randomness) in enumerate(games_n_rules, start=1):
        print(f"[{i}/{len(games_n_rules)}] {game} (n_rules={n_rules}, random={has_randomness})")
        cfg = RunConfig(
            game=game,
            level=args.level,
            max_episode_steps=args.max_episode_steps,
            vmap=True,
        )
        result = _profile_one_game(game, n_rules, has_randomness, cfg, n_envs=args.n_envs)
        results.append(result)
        if result["status"] == "ok":
            print(
                f"  compile={result['compile_seconds']:.3f}s "
                f"first_exec={result['first_execute_seconds']:.3f}s "
                f"second_exec={result['second_execute_seconds']:.3f}s"
            )
        else:
            print(f"  ERROR: {result['error']}")

    summary = _summarize(results, top_n=args.top_n)
    payload = {
        "config": vars(args),
        "device": devices[0].device_kind,
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved profiling results to {args.output_path}")
    print(
        f"Success={summary['n_success']} Errors={summary['n_errors']} "
        f"Median compile={summary.get('compile_seconds_median', float('nan')):.3f}s"
    )
    print("\nSlowest compile games:")
    for row in summary["slowest_compile_games"][: args.top_n]:
        print(
            f"  {row['game']}: compile={row['compile_seconds']:.3f}s "
            f"(n_rules={row['n_rules']}, random={row['has_randomness']})"
        )


if __name__ == "__main__":
    main()
