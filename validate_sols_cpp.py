"""Validate the C++ PuzzleScript engine against NodeJS reference solutions."""
import glob
import json
import math
import os
import sys
import traceback
from collections import OrderedDict
from typing import List, Optional

import hydra
import imageio
import numpy as np
import submitit
from lark import Lark

# Ensure the project root is on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conf.config import CppValidationConfig
from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.globals import CPP_VALIDATED_JS_SOLS_DIR, DATA_DIR, JS_SOLS_DIR, LARK_SYNTAX_PATH
from puzzlescript_jax.preprocessing import SIMPLIFIED_GAMES_DIR, get_tree_from_txt
from puzzlescript_jax.utils import get_list_of_games_for_testing
from puzzlescript_nodejs.utils import replay_actions_js


def _dedupe_preserve_order(items, key_fn=None):
    seen = OrderedDict()
    for item in items:
        key = key_fn(item) if key_fn is not None else item
        seen[key] = item
    return list(seen.values())


def _make_renderer(js_engine, compiled_json):
    from puzzlescript_cpp import Renderer

    renderer = Renderer()
    renderer.load_sprite_data(str(js_engine.serializeSpriteDataJSON()))
    renderer.load_render_config(compiled_json)
    return renderer


def _make_side_by_side_frames(left_frames, right_frames, separator_w=2):
    if len(left_frames) == 0 or len(right_frames) == 0:
        return []

    left_frames = [np.asarray(frame, dtype=np.uint8) for frame in left_frames]
    right_frames = [np.asarray(frame, dtype=np.uint8) for frame in right_frames]
    max_len = max(len(left_frames), len(right_frames))
    while len(left_frames) < max_len:
        left_frames.append(left_frames[-1])
    while len(right_frames) < max_len:
        right_frames.append(right_frames[-1])

    combo_frames = []
    for left_frame, right_frame in zip(left_frames, right_frames):
        left_h, _ = left_frame.shape[:2]
        right_h, _ = right_frame.shape[:2]
        frame_h = max(left_h, right_h)
        if left_h < frame_h:
            left_frame = np.pad(left_frame, ((0, frame_h - left_h), (0, 0), (0, 0)))
        if right_h < frame_h:
            right_frame = np.pad(right_frame, ((0, frame_h - right_h), (0, 0), (0, 0)))
        separator = np.full((frame_h, separator_w, 3), 128, dtype=np.uint8)
        combo_frames.append(np.concatenate([left_frame, separator, right_frame], axis=1))

    return combo_frames


def make_gif(frames, gif_path, scale=1, duration=0.5):
    if scale > 1:
        frames = [
            np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
            for frame in frames
        ]
    gif_dir = os.path.dirname(gif_path)
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)


def compile_game_for_cpp(js_engine, parser, game):
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}.txt")
    if not os.path.isfile(game_path):
        get_tree_from_txt(parser=parser, game=game, test_env_init=False, overwrite=True)
    simplified_path = f"{game_path[:-4]}_simplified.txt"
    with open(simplified_path, "r") as f:
        game_text = f.read()
    js_engine.compile(["restart"], game_text)
    json_str = str(js_engine.serializeCompiledStateJSON())
    return game_text, json_str


def replay_actions_cpp(cpp_engine, serialized_json, actions, level_i):
    cpp_engine.load_from_json(serialized_json)
    cpp_engine.load_level(level_i)
    cpp_states = []
    cpp_winning = []
    max_again = 50

    cpp_states.append(list(cpp_engine.get_objects()))
    cpp_winning.append(bool(cpp_engine.is_winning()))

    for action in actions:
        cpp_engine.process_input(action)
        again_steps = 0
        while cpp_engine.is_againing() and again_steps < max_again:
            cpp_engine.process_input(-1)
            again_steps += 1
        cpp_states.append(list(cpp_engine.get_objects()))
        cpp_winning.append(bool(cpp_engine.is_winning()))

    return cpp_states, cpp_winning


def choose_level_solution_paths(sol_dir):
    level_sols = glob.glob(os.path.join(sol_dir, "*level-*.json"))
    level_sols = [os.path.basename(path) for path in level_sols]
    level_ints = [int(os.path.basename(path).split("-")[-1].split(".")[0]) for path in level_sols]
    sorted_idxs = np.argsort(level_ints)
    level_ints = [level_ints[i] for i in sorted_idxs]
    level_sols = [level_sols[i] for i in sorted_idxs]

    level_ints_to_sols = {}
    for level_i, sol_name in zip(level_ints, level_sols):
        level_ints_to_sols.setdefault(level_i, []).append(sol_name)

    chosen_sols = []
    chosen_level_ints = []
    for level_i, sols in level_ints_to_sols.items():
        bfs_sols = [sol for sol in sols if "solveBFS" in sol]
        astar_sols = [sol for sol in sols if "solveAStar" in sol]
        mcts_sols = [sol for sol in sols if "solveMCTS" in sol]

        if bfs_sols:
            sols_to_consider = bfs_sols
        elif astar_sols:
            sols_to_consider = astar_sols
        elif mcts_sols:
            sols_to_consider = mcts_sols
        else:
            sols_to_consider = sols

        if len(sols_to_consider) == 1:
            chosen_sol = sols_to_consider[0]
        else:
            n_steps = [
                int(os.path.basename(path).split("-steps_")[0].split("_")[-1])
                if "-steps_" in os.path.basename(path)
                else 10_000
                for path in sols_to_consider
            ]
            chosen_sol = sols_to_consider[int(np.argmax(n_steps))]

        chosen_level_ints.append(level_i)
        chosen_sols.append(chosen_sol)

    return chosen_level_ints, chosen_sols


def is_runtime_timeout_log(log: str) -> bool:
    if not log:
        return False
    log_lower = log.lower()
    return "timed out" in log_lower or "timed-out" in log_lower or "timeout" in log_lower


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_stats(results, val_results_path, games, n_levels, n_success, n_compile_error, n_timeout_error,
               n_runtime_error, n_state_error, n_win_error, n_unvalidated_levels):
    results["stats"]["total_games"] = len(games)
    results["stats"]["total_levels"] = n_levels
    results["stats"]["successful_solutions"] = n_success
    results["stats"]["compile_error"] = n_compile_error
    results["stats"]["timeout"] = n_timeout_error
    results["stats"]["runtime_error"] = n_runtime_error
    results["stats"]["state_error"] = n_state_error
    results["stats"]["win_error"] = n_win_error
    results["stats"]["unvalidated_levels"] = n_unvalidated_levels
    results["stats"]["valid_games"] = len(results["valid_games"])
    results["stats"]["partial_valid_games"] = len(results["partial_valid_games"])

    with open(val_results_path, "w") as f:
        json.dump(results, f, indent=2)


@hydra.main(version_base="1.3", config_path="conf", config_name="cpp_validation_config")
def main_launch(cfg: CppValidationConfig):
    if cfg.slurm:
        games = get_list_of_games_for_testing(dataset=cfg.dataset)
        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "validate_sols_cpp"))
        executor.update_parameters(
            slurm_job_name="validate_sols_cpp",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: CppValidationConfig, games: Optional[List[str]] = None):
    import puzzlescript_cpp._puzzlescript_cpp as ps

    if cfg.slurm:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    backend = NodeJSPuzzleScriptBackend()
    js_engine = backend.engine
    cpp_engine = ps.Engine()

    with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
        grammar = f.read()
    parser = Lark(grammar, start="ps_game", maybe_placeholders=False)

    if games is not None:
        games = games
    elif cfg.game is None:
        games = get_list_of_games_for_testing(dataset=cfg.dataset, random_order=cfg.random_order)
    else:
        games = [cfg.game]

    current_game_names = {
        os.path.splitext(game_file)[0]
        for game_file in os.listdir(os.path.join(DATA_DIR, "scraped_games"))
    }
    games = [game for game in games if game in current_game_names]
    games = _dedupe_preserve_order(games)
    if not cfg.include_test_games:
        games = [game for game in games if not game.startswith("test_")]

    output_dir = cfg.output_dir or CPP_VALIDATED_JS_SOLS_DIR
    os.makedirs(output_dir, exist_ok=True)

    val_results_path = os.path.join("data", "cpp_validation_results.json")
    results = {
        "stats": {},
        "compile_error": [],
        "timeout": [],
        "runtime_error": {},
        "state_error": {},
        "win_error": {},
        "success": {},
        "valid_games": [],
        "partial_valid_games": [],
    }

    n_levels = 0
    n_compile_error = 0
    n_timeout_error = 0
    n_runtime_error = 0
    n_state_error = 0
    n_win_error = 0
    n_success = 0
    n_unvalidated_levels = 0

    for game in games:
        game_name = os.path.basename(game)
        sol_dir = os.path.join(JS_SOLS_DIR, game)
        cpp_sol_dir = os.path.join(output_dir, game)
        os.makedirs(cpp_sol_dir, exist_ok=True)

        compile_log_path = os.path.join(cpp_sol_dir, "compile_err.txt")
        if cfg.overwrite and os.path.exists(compile_log_path):
            os.remove(compile_log_path)
        if os.path.exists(compile_log_path) and not cfg.overwrite:
            with open(compile_log_path, "r") as f:
                compile_log = f.read()
            if cfg.aggregate:
                if is_runtime_timeout_log(compile_log):
                    results["timeout"].append({"game": game_name, "log": compile_log})
                    n_timeout_error += 1
                else:
                    results["compile_error"].append({"game": game_name, "log": compile_log})
                    n_compile_error += 1
            else:
                if is_runtime_timeout_log(compile_log):
                    n_timeout_error += 1
                else:
                    n_compile_error += 1
            n_levels += 1
            print(f"Skipping {game_name} because compile error log already exists")
            continue

        level_ints, level_sols = choose_level_solution_paths(sol_dir)
        if len(level_sols) == 0:
            print(f"No JS solutions found for {game_name}")
            continue

        game_success = True
        game_partial_success = False
        game_compile_error = False

        if not cfg.aggregate:
            try:
                game_text, serialized_json = compile_game_for_cpp(js_engine, parser, game)
            except Exception:
                compile_log = traceback.format_exc()
                with open(compile_log_path, "w") as f:
                    f.write(compile_log)
                if is_runtime_timeout_log(compile_log):
                    n_timeout_error += 1
                else:
                    n_compile_error += 1
                game_success = False
                game_compile_error = True
                print(f"Compile error for {game_name}")
                continue

            cpp_renderer = None
            n_objs = None
            if cfg.render:
                try:
                    cpp_renderer = _make_renderer(js_engine, serialized_json)
                    cpp_engine.load_from_json(serialized_json)
                    cpp_engine.load_level(0)
                    n_objs = cpp_engine.get_object_count()
                except Exception:
                    cpp_renderer = None
                    print(f"Warning: failed to initialize renderer for {game_name}")
                    traceback.print_exc()

        print(f"Processing {game_name}")
        for level_i, level_sol_name in zip(level_ints, level_sols):
            if game_compile_error:
                break

            n_levels += 1
            level_sol_path = os.path.join(sol_dir, level_sol_name)
            state_log_path = os.path.join(cpp_sol_dir, f"level-{level_i}_state_err.txt")
            win_log_path = os.path.join(cpp_sol_dir, f"level-{level_i}_win_err.txt")
            run_log_path = os.path.join(cpp_sol_dir, f"level-{level_i}_runtime_err.txt")
            success_path = os.path.join(cpp_sol_dir, f"level-{level_i}_success.json")
            js_gif_path = os.path.join(cpp_sol_dir, f"level-{level_i}_js.gif")
            cpp_gif_path = os.path.join(cpp_sol_dir, f"level-{level_i}_cpp.gif")
            compare_gif_path = os.path.join(cpp_sol_dir, f"level-{level_i}_compare.gif")

            existing_paths = [
                state_log_path,
                win_log_path,
                run_log_path,
                success_path,
                js_gif_path,
                cpp_gif_path,
                compare_gif_path,
            ]
            if any(os.path.exists(path) for path in existing_paths):
                if cfg.overwrite:
                    for path in existing_paths:
                        if os.path.exists(path):
                            os.remove(path)
                else:
                    if os.path.exists(run_log_path):
                        with open(run_log_path, "r") as f:
                            run_log = f.read()
                        if cfg.aggregate:
                            if is_runtime_timeout_log(run_log):
                                results["timeout"].append({"game": game_name, "level": level_i, "log": run_log})
                                n_timeout_error += 1
                            else:
                                results["runtime_error"].setdefault(game_name, []).append(
                                    {"level": level_i, "log": run_log}
                                )
                                n_runtime_error += 1
                        else:
                            if is_runtime_timeout_log(run_log):
                                n_timeout_error += 1
                            else:
                                n_runtime_error += 1
                        game_success = False
                    elif os.path.exists(state_log_path):
                        if cfg.aggregate:
                            with open(state_log_path, "r") as f:
                                state_log = f.read()
                            results["state_error"].setdefault(game_name, []).append(
                                {"level": level_i, "log": state_log}
                            )
                        n_state_error += 1
                        game_success = False
                    elif os.path.exists(win_log_path):
                        if cfg.aggregate:
                            with open(win_log_path, "r") as f:
                                win_log = f.read()
                            results["win_error"].setdefault(game_name, []).append(
                                {"level": level_i, "log": win_log}
                            )
                        n_win_error += 1
                        game_success = False
                    elif os.path.exists(success_path):
                        if cfg.aggregate:
                            with open(success_path, "r") as f:
                                success_entry = json.load(f)
                            results["success"].setdefault(game_name, []).append(success_entry)
                        n_success += 1
                        game_partial_success = True
                    elif (
                        os.path.exists(compare_gif_path)
                        or os.path.exists(cpp_gif_path)
                        or os.path.exists(js_gif_path)
                    ):
                        if cfg.aggregate:
                            success_entry = {"level": level_i}
                            results["success"].setdefault(game_name, []).append(success_entry)
                        n_success += 1
                        game_partial_success = True
                    print(f"Skipping level {level_i} because validation artifacts already exist")
                    continue

            if cfg.aggregate:
                n_unvalidated_levels += 1
                game_success = False
                continue

            try:
                with open(level_sol_path, "r") as f:
                    sol_data = json.load(f)

                actions = sol_data.get("actions", sol_data.get("sol", sol_data.get("solution", [])))
                if isinstance(actions, dict):
                    actions = actions.get("actions", [])

                _, replayed_js_states, js_winning = replay_actions_js(
                    js_engine,
                    backend.solver,
                    actions,
                    game_text,
                    level_i,
                    stop_on_win=False,
                    return_winning=True,
                )
                js_states = [list(level["dat"]) for level in replayed_js_states]
                cpp_states, cpp_winning = replay_actions_cpp(cpp_engine, serialized_json, actions, level_i)

                mismatch_step = None
                for step in range(min(len(js_states), len(cpp_states))):
                    if js_states[step] != cpp_states[step]:
                        mismatch_step = step
                        break
                if mismatch_step is None and len(js_states) != len(cpp_states):
                    mismatch_step = min(len(js_states), len(cpp_states))

                if mismatch_step is not None:
                    with open(state_log_path, "w") as f:
                        f.write(f"Level {level_i} state mismatch\n")
                        f.write(f"Actions: {actions}\n")
                        f.write(f"Mismatch step: {mismatch_step}\n")
                        if mismatch_step < len(js_states):
                            f.write(f"JS state: {js_states[mismatch_step]}\n")
                        if mismatch_step < len(cpp_states):
                            f.write(f"CPP state: {cpp_states[mismatch_step]}\n")
                    n_state_error += 1
                    game_success = False
                    print(f"State mismatch for {game_name} level {level_i} at step {mismatch_step}")
                elif js_winning != cpp_winning:
                    with open(win_log_path, "w") as f:
                        f.write(f"Level {level_i} win mismatch\n")
                        f.write(f"Actions: {actions}\n")
                        f.write(f"JS winning: {js_winning}\n")
                        f.write(f"CPP winning: {cpp_winning}\n")
                    n_win_error += 1
                    game_success = False
                    print(f"Win mismatch for {game_name} level {level_i}")
                else:
                    success_entry = {"level": level_i, "status": "success"}
                    write_json(success_path, success_entry)
                    n_success += 1
                    game_partial_success = True
                    print(f"Level {level_i} succeeded")

                should_render = (
                    cfg.render
                    and cpp_renderer is not None
                    and (not cfg.render_mismatches_only or mismatch_step is not None or js_winning != cpp_winning)
                )
                if should_render:
                    cpp_engine.load_from_json(serialized_json)
                    cpp_engine.load_level(level_i)
                    width = cpp_engine.get_width()
                    height = cpp_engine.get_height()
                    cpp_renderer.reset_viewport(width, height)
                    print(f"Rendering level {level_i} with grid size {width}x{height} and {n_objs} objects in JS engine.")
                    backend.render_gif(
                        game_text=game_text,
                        level_i=level_i,
                        actions=actions,
                        gif_path=js_gif_path,
                        frame_duration_s=0.3,
                        scale=1,
                    )
                    js_frames = [
                        np.asarray(frame, dtype=np.uint8)[..., :3]
                        for frame in imageio.mimread(js_gif_path)
                    ]
                    print(f"Rendering in C++ engine.")
                    cpp_frames = [
                        cpp_renderer.render_objects(np.array(state, dtype=np.int32), width, height, n_objs)
                        for state in cpp_states
                    ]
                    compare_frames = _make_side_by_side_frames(js_frames, cpp_frames)
                    make_gif(cpp_frames, cpp_gif_path, scale=1, duration=0.5)
                    if compare_frames:
                        make_gif(compare_frames, compare_gif_path, scale=1, duration=0.5)
                    print(f"Rendered level {level_i} GIFs to {cpp_sol_dir}")

            except KeyboardInterrupt:
                raise
            except Exception:
                err_log = traceback.format_exc()
                with open(run_log_path, "w") as f:
                    f.write(err_log)
                if is_runtime_timeout_log(err_log):
                    n_timeout_error += 1
                else:
                    n_runtime_error += 1
                game_success = False
                print(f"Runtime error for {game_name} level {level_i}")

        if cfg.aggregate:
            if game_success:
                results["valid_games"].append({"game": game_name})
            elif game_partial_success:
                results["partial_valid_games"].append({"game": game_name})

    if cfg.aggregate:
        save_stats(
            results,
            val_results_path,
            games,
            n_levels,
            n_success,
            n_compile_error,
            n_timeout_error,
            n_runtime_error,
            n_state_error,
            n_win_error,
            n_unvalidated_levels,
        )
        print(f"Validation results saved to {val_results_path}")

    return results


if __name__ == "__main__":
    main_launch()
