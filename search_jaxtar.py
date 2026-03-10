"""Run JAXtar search on PuzzleJAX (PuzzleScript) environments and log results.

Usage examples:
    # Single game, single level
    python profile_jaxtar.py --game blocks --level 0

    # Single game, all levels
    python profile_jaxtar.py --game blocks

    # All priority games
    python profile_jaxtar.py

    # All games with custom search budget
    python profile_jaxtar.py --all_games -m 500000 -b 5000

    # Beam search instead of A*
    python profile_jaxtar.py --game blocks --algo beam -m 1000000

Results are saved to data/jaxtar_sols/<GAME>/<algo>_<max_nodes>-nodes_level-<i>.json
GIFs are saved alongside with the same base filename.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

# JAXtar modules are importable when JAXtar/ is on sys.path.
JAXTAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JAXtar")
if JAXTAR_DIR not in sys.path:
    sys.path.insert(0, JAXTAR_DIR)

from puzzlescript_jax.globals import DATA_DIR, GAMES_N_LEVELS_PATH
from puzzlescript_jax.utils import get_list_of_games_for_testing

from puzzlescript_jax.wrappers import PuzzleJaxPuxleEnv, PuzzleJaxHeuristic

from JAxtar.stars.astar import astar_builder
from JAxtar.stars.search_base import SearchResult
from JAxtar.beamsearch.heuristic_beam import beam_builder
from helpers.visualization import (
    PathStep,
    build_path_steps_from_actions,
)
from exit_train import extract_best_env_f_path

JAXTAR_SOLS_DIR = os.path.join(DATA_DIR, "jaxtar_sols")


def get_n_levels(game: str) -> int:
    """Return the number of levels for *game* using the cached JSON, falling back
    to initialising a temporary env to count them."""
    if os.path.isfile(GAMES_N_LEVELS_PATH):
        with open(GAMES_N_LEVELS_PATH, "r") as f:
            n_levels_map = json.load(f)
        if game in n_levels_map:
            return int(n_levels_map[game])
    # Fallback: instantiate env for level 0 and count levels
    try:
        puzzle = PuzzleJaxPuxleEnv(game=game, level_i=0)
        return len(puzzle.env.levels)
    except Exception:
        return 1


def save_gif(path_steps: List[PathStep], solve_config, gif_path: str,
             max_animation_time: float = 10.0):
    """Render a GIF from a list of PathSteps and save to *gif_path*."""
    import imageio

    imgs = []
    path_states = [step.state for step in path_steps]
    for idx, step in enumerate(path_steps):
        img = step.state.img(idx=idx, path=path_states, solve_config=solve_config)
        imgs.append(img)
    if not imgs:
        return
    fps = 4
    num_frames = len(imgs)
    if num_frames / fps > max_animation_time:
        fps = num_frames / max_animation_time
    imageio.mimsave(gif_path, imgs, fps=fps)


def run_search_on_level(
    game: str,
    level_i: int,
    algo: str,
    max_nodes: int,
    batch_size: int,
    cost_weight: float,
    render_gif: bool,
    overwrite: bool,
    max_animation_time: float,
) -> Optional[dict]:
    """Run JAXtar search on a single game level. Returns the result dict or None on error."""
    algo_tag = algo
    game_sols_dir = os.path.join(JAXTAR_SOLS_DIR, game)
    os.makedirs(game_sols_dir, exist_ok=True)

    base_name = f"{algo_tag}_{max_nodes}-nodes_level-{level_i}"
    json_path = os.path.join(game_sols_dir, f"{base_name}.json")
    gif_path = os.path.join(game_sols_dir, f"{base_name}.gif")

    if not overwrite and os.path.isfile(json_path):
        print(f"  Already solved, skipping. ({json_path})")
        return None

    # ---------- build puzzle + heuristic ----------
    try:
        puzzle = PuzzleJaxPuxleEnv(game=game, level_i=level_i)
    except Exception as e:
        print(f"  Error initialising puzzle for {game} level {level_i}: {e}")
        traceback.print_exc()
        return {"error": str(e)}

    heuristic = PuzzleJaxHeuristic(puzzle)

    # Align max_nodes to batch_size
    effective_max_nodes = (max_nodes // batch_size) * batch_size
    if effective_max_nodes == 0:
        effective_max_nodes = batch_size

    # ---------- build search function ----------
    if algo == "astar":
        search_fn = astar_builder(
            puzzle,
            heuristic,
            batch_size=batch_size,
            max_nodes=effective_max_nodes,
            cost_weight=cost_weight,
        )
    elif algo == "beam":
        search_fn = beam_builder(
            puzzle,
            heuristic,
            batch_size=batch_size,
            max_nodes=effective_max_nodes,
            cost_weight=cost_weight,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # ---------- run search ----------
    solve_config, init_state = puzzle.get_inits(jax.random.PRNGKey(0))

    # Warmup: the first call triggers JIT compilation (which dominates wall time).
    warmup_start = time.time()
    warmup_result = search_fn(solve_config, init_state)
    warmup_result.solved.block_until_ready()
    compile_time = time.time() - warmup_start
    print(f"  Warmup (compile + execute): {compile_time:.2f}s")

    # Timed run: JIT cache is now warm, so this measures pure search speed.
    start = time.time()
    search_result = search_fn(solve_config, init_state)
    solved = bool(search_result.solved.block_until_ready())
    elapsed = time.time() - start

    compile_time_est = compile_time - elapsed

    generated = int(search_result.generated_size)
    states_per_sec = generated / elapsed if elapsed > 0 else 0

    print(f"  {'SOLVED' if solved else 'UNSOLVED'} | "
          f"{generated:,} states | {elapsed:.2f}s | {states_per_sec:,.0f} states/s "
          f"(est. compile: {compile_time_est:.2f}s)")

    # ---------- extract solution ----------
    actions_list: list = []
    cost_val = float("inf")
    path_steps: Optional[List[PathStep]] = None

    if solved:
        solved_idx = search_result.solved_idx
        cost_val = float(search_result.get_cost(solved_idx))

        # Try to extract actions via the method available on the result type
        try:
            if hasattr(search_result, "solution_trace"):
                states_trace, costs_trace, dists_trace, actions_trace = search_result.solution_trace()
                actions_list = [int(a) for a in actions_trace]
                if render_gif:
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=init_state,
                        actions=actions_trace,
                        heuristic=heuristic,
                        states=states_trace if states_trace else None,
                        costs=costs_trace if costs_trace else None,
                        dists=dists_trace if dists_trace else None,
                    )
            elif hasattr(search_result, "solution_actions"):
                actions_trace = search_result.solution_actions()
                actions_list = [int(a) for a in actions_trace]
                if render_gif:
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=init_state,
                        actions=actions_trace,
                        heuristic=heuristic,
                    )
            else:
                path = search_result.get_solved_path()
                actions_list = []
                for node in path:
                    action = getattr(node, "action", None)
                    if action is not None:
                        actions_list.append(int(action))
                if render_gif:
                    # Use build_path_steps_from_actions (replaying actions from
                    # the initial state) instead of build_path_steps_from_nodes
                    # (which looks up states by hash-table slot index).  Cuckoo
                    # hashing can relocate states during insertion, leaving
                    # stale slot indices in the parent pointers and causing the
                    # GIF to show impossible state transitions.
                    path_steps = build_path_steps_from_actions(
                        puzzle=puzzle,
                        solve_config=solve_config,
                        initial_state=init_state,
                        actions=actions_list,
                        heuristic=heuristic,
                    )
        except Exception as e:
            print(f"  Warning: could not extract solution path: {e}")
            traceback.print_exc()

    # ---------- save GIF ----------
    if render_gif and path_steps is not None and len(path_steps) > 0:
        try:
            save_gif(path_steps, solve_config, gif_path, max_animation_time)
            print(f"  GIF saved to {gif_path}")
        except Exception as e:
            print(f"  Warning: GIF rendering failed: {e}")
            traceback.print_exc()

    # ---------- extract best-f path (env heuristic) ----------
    best_f_info = None
    try:
        best_f_info = extract_best_env_f_path(
            puzzle, search_result, solve_config, init_state,
            rule_heuristic=heuristic, render=render_gif,
        )
        if best_f_info is not None:
            print(f"  Best env-f: g={best_f_info['best_g']:.1f}  "
                  f"h_env={best_f_info['best_env_h']:.1f}  "
                  f"f_env={best_f_info['best_env_f']:.1f}  "
                  f"path_len={len(best_f_info['best_actions'])}")
            # Render GIF of best-f path
            if (render_gif and best_f_info["path_steps"] is not None
                    and len(best_f_info["path_steps"]) > 0):
                best_f_gif_path = os.path.join(
                    game_sols_dir, f"{base_name}_best_env_f.gif"
                )
                try:
                    save_gif(
                        best_f_info["path_steps"], solve_config,
                        best_f_gif_path, max_animation_time,
                    )
                    print(f"  Best-f GIF saved to {best_f_gif_path}")
                except Exception as e:
                    print(f"  Warning: Best-f GIF render failed: {e}")
    except Exception as e:
        print(f"  Warning: best-f extraction failed: {e}")
        traceback.print_exc()

    # ---------- save JSON ----------
    result_dict = {
        "won": solved,
        "actions": actions_list,
        "cost": cost_val if math.isfinite(cost_val) else None,
        "generated_states": generated,
        "time": elapsed,
        "compile_time": compile_time,
        "compile_time_est": compile_time_est,
        "states_per_sec": states_per_sec,
        "algo": algo,
        "max_nodes": max_nodes,
        "batch_size": batch_size,
        "cost_weight": cost_weight,
        # Best-f diagnostics (environment heuristic)
        "best_env_h": best_f_info["best_env_h"] if best_f_info else None,
        "best_env_g": best_f_info["best_g"] if best_f_info else None,
        "best_env_f": best_f_info["best_env_f"] if best_f_info else None,
        "best_env_path_len": len(best_f_info["best_actions"]) if best_f_info else None,
        "best_env_actions": best_f_info["best_actions"] if best_f_info else None,
    }

    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"  JSON saved to {json_path}")

    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run JAXtar search on PuzzleScript games and log results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--game", type=str, default=None,
                        help="Single game name to solve. If omitted, runs priority games.")
    parser.add_argument("--level", type=int, default=None,
                        help="Single level index. If omitted, runs all levels of each game.")
    parser.add_argument("--all_games", action="store_true",
                        help="Run on all available games (not just priority list).")
    parser.add_argument("--algo", type=str, default="astar", choices=["astar", "beam"],
                        help="Search algorithm (default: astar).")
    parser.add_argument("-m", "--max_nodes", type=int, default=1_000_000,
                        help="Maximum number of search nodes (default: 1000000).")
    parser.add_argument("-b", "--batch_size", type=int, default=10_000,
                        help="Batch size for search (default: 10000).")
    parser.add_argument("-w", "--cost_weight", type=float, default=0.6,
                        help="Cost weight for A*/beam f = g + w*h (default: 0.6).")
    parser.add_argument("--no_gif", action="store_true",
                        help="Skip GIF rendering.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results.")
    parser.add_argument("--max_animation_time", type=float, default=10.0,
                        help="Max GIF animation duration in seconds (default: 10).")
    parser.add_argument("--include_randomness", action="store_true",
                        help="Include games with randomness.")
    parser.add_argument("--random_order", action="store_true",
                        help="Shuffle game order.")

    args = parser.parse_args()

    # ---------- determine game list ----------
    if args.game is not None:
        games = [args.game]
    else:
        games = get_list_of_games_for_testing(
            all_games=args.all_games,
            include_random=args.include_randomness,
            random_order=args.random_order,
        )

    os.makedirs(JAXTAR_SOLS_DIR, exist_ok=True)
    render_gif = not args.no_gif

    total_solved = 0
    total_levels = 0

    for game in games:
        print(f"\n{'='*60}")
        print(f"Game: {game}")
        print(f"{'='*60}")

        if args.level is not None:
            levels = [args.level]
        else:
            n_levels = get_n_levels(game)
            levels = list(range(n_levels))

        for level_i in levels:
            total_levels += 1
            print(f"\n  Level {level_i}:")
            try:
                result = run_search_on_level(
                    game=game,
                    level_i=level_i,
                    algo=args.algo,
                    max_nodes=args.max_nodes,
                    batch_size=args.batch_size,
                    cost_weight=args.cost_weight,
                    render_gif=render_gif,
                    overwrite=args.overwrite,
                    max_animation_time=args.max_animation_time,
                )
                if result is not None and result.get("won", False):
                    total_solved += 1
            except Exception as e:
                print(f"  FATAL error on {game} level {level_i}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done. Solved {total_solved}/{total_levels} levels.")
    print(f"Results saved to {JAXTAR_SOLS_DIR}")


if __name__ == "__main__":
    main()
