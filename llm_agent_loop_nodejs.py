import argparse
import multiprocessing
import os
import string
import json
from dataclasses import dataclass

from lark import Lark

from LLM_agent import LLMGameAgent
from llm_agent_loop import (
    CUSTOM_GAMES_DIR,
    STATE_FILE_BASENAME,
    check_run_file_exists,
    claim_next_job,
    extract_section,
    get_run_file_path,
    list_step_log_files,
    parse_legend,
    parse_step_log_file,
    release_job,
)
from puzzlejax.backends.nodejs import NodeJSPuzzleScriptBackend
from puzzlejax.globals import LARK_SYNTAX_PATH, PRIORITY_GAMES
from puzzlejax.utils import level_to_int_arr


ACTION_SPACE = [0, 1, 2, 3, 4]
ACTION_MEANINGS = {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}


def _available_render_chars():
    chars = [c for c in string.printable if (not c.isspace()) and c not in {"`"}]
    extras = [
        "■", "□", "▲", "△", "●", "○", "◆", "◇", "★", "☆",
        "♠", "♣", "♥", "♦", "☉", "☼", "☯", "☒", "☐", "☑",
        "◉", "◎", "◍", "◌", "◐", "◑", "◒", "◓", "◔", "◕",
    ]
    for ch in extras:
        if ch not in chars:
            chars.append(ch)
    return chars


@dataclass
class NodeJSAsciiRenderer:
    legend_mapping: dict[str, list[str]]

    def __post_init__(self):
        self.combo_to_char: dict[tuple[str, ...], str] = {}
        self.available_chars = _available_render_chars()
        self._seed_chars_from_legend()

    def _seed_chars_from_legend(self):
        for char, objs in self.legend_mapping.items():
            if len(char) != 1 or char.isspace():
                continue
            combo = tuple(obj for obj in objs if obj != "background")
            if not combo:
                combo = ("background",)
            self.combo_to_char.setdefault(combo, char)
            if char in self.available_chars:
                self.available_chars.remove(char)

    def _char_for_combo(self, combo: tuple[str, ...]) -> str:
        if combo not in self.combo_to_char:
            if not self.available_chars:
                raise ValueError("Ran out of characters while rendering NodeJS PuzzleScript state.")
            self.combo_to_char[combo] = self.available_chars.pop(0)
        return self.combo_to_char[combo]

    def render(self, backend: NodeJSPuzzleScriptBackend, level, objs: list[str]) -> str:
        level_arr = level_to_int_arr(level, len(objs))
        mini, minj, maxi, maxj = backend._get_visible_bounds(level)
        used_chars: list[str] = []
        seen_chars: set[str] = set()
        rows: list[str] = []

        for y in range(minj, maxj):
            row_chars = []
            for x in range(mini, maxi):
                cell_bits = int(level_arr[x, y])
                names = [objs[obj_i] for obj_i in range(len(objs)) if ((cell_bits >> obj_i) & 1) == 1]
                combo = tuple(name for name in names if name != "background")
                if not combo:
                    combo = ("background",)
                char = self._char_for_combo(combo)
                if char not in seen_chars:
                    seen_chars.add(char)
                    used_chars.append(char)
                row_chars.append(char)
            rows.append("".join(row_chars))

        legend_lines = []
        for char in used_chars:
            combo = next(combo for combo, combo_char in self.combo_to_char.items() if combo_char == char)
            legend_lines.append(f"{char}: {', '.join(combo)}")

        return f"LEGEND:\n" + "\n".join(legend_lines) + "\n\nMAP:\n" + "\n".join(rows)


def collect_game_info(game_name, start_level):
    game_path = os.path.join(CUSTOM_GAMES_DIR, f"{game_name}.txt")
    if not os.path.exists(game_path):
        print(f"Error: Game file not found at {game_path}. Skipping game.")
        return None

    rules_lines = extract_section(game_path, "RULES")
    legend_lines = extract_section(game_path, "LEGEND")
    rules = "\n".join(rules_lines)
    legend_mapping = parse_legend(legend_lines)
    if not rules:
        print(f"Error: Missing rules for game {game_name}. Skipping game.")
        return None

    try:
        with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
            puzzlescript_grammar = f.read()
        grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
        backend = NodeJSPuzzleScriptBackend()
        game_text = backend.compile_game(grammar_parser, game_name)
        num_levels = backend.get_num_levels()
        backend.unload_game()
    except Exception as e:
        print(f"Error collecting NodeJS game info for {game_name}: {type(e).__name__}, {e}")
        return None

    if num_levels <= 0:
        print(f"Error: No levels found for game '{game_name}'. Skipping game.")
        return None

    levels_to_process = list(range(start_level, num_levels))
    if not levels_to_process:
        print(f"Warning: start level {start_level} is past the last level for '{game_name}'. Skipping game.")
        return None

    return {
        "game_name": game_name,
        "game_path": game_path,
        "game_text": game_text,
        "rules": rules,
        "legend_mapping": legend_mapping,
        "num_levels": num_levels,
        "levels_to_process": levels_to_process,
    }


def process_game_level(agent, game_info, level_index, run_id, save_dir, model,
                       max_steps, think_aloud: bool, memory: int, force=False):
    game_name = game_info["game_name"]
    current_run_filepath = get_run_file_path(save_dir, model, game_name, run_id,
                                             level_index, think_aloud, memory)
    current_run_logs_dir = current_run_filepath[:-5] + "_logs"
    os.makedirs(current_run_logs_dir, exist_ok=True)

    run_file_exists = os.path.exists(current_run_filepath)
    if run_file_exists and not force:
        print(f"Run {run_id} for Game: {game_name}, Level: {level_index} already exists. Skipping.")
        return True

    try:
        with open(current_run_filepath, "w", encoding="utf-8") as f:
            f.write('{"status": "running"}')
    except Exception as e:
        print(f"Failed to create lock file for {current_run_filepath}: {e}")
        return False

    try:
        if level_index < 0 or level_index >= game_info["num_levels"]:
            print(f"Warning: Invalid level_index {level_index} for game '{game_name}'. Skipping level.")
            return False

        backend = NodeJSPuzzleScriptBackend()
        backend.load_level(game_info["game_text"], level_index)
        backend.solver.precalcDistances(backend.engine)
        renderer = NodeJSAsciiRenderer(game_info["legend_mapping"])
        rules = game_info["rules"]

        level = backend.engine.backupLevel()
        objs = list(backend.engine.getState().idDict)
        score = float(backend.solver.getScore(backend.engine))
        won = bool(backend.engine.getWinning())
        ascii_state = renderer.render(backend, level, objs)

        print(f"\n=== Processing Game: {game_name}, Level: {level_index}, Run: {run_id} ===")

        result = {
            "model": model,
            "game": game_name,
            "level": level_index,
            "run": run_id,
            "win": won,
            "initial_ascii": ascii_state.split("\n"),
            "action_sequence": [],
            "reward_sequence": [],
            "heuristic_sequence": [],
            "state_data": {
                "score": score,
                "win": won,
                "step": 0,
            },
        }
        action_sequence_for_render = []

        state_history_lst = []
        replayed_steps = 0

        if not force:
            step_files = list_step_log_files(current_run_logs_dir)
            if step_files and (not run_file_exists):
                print(f"Attempting resume from {len(step_files)} log steps in {current_run_logs_dir}")
                replay_ok = True
                parsed_steps = []
                for _, step_file in step_files:
                    try:
                        parsed_steps.append(parse_step_log_file(step_file, ACTION_SPACE))
                    except Exception as e:
                        replay_ok = False
                        print(f"Resume parse failed for {step_file}: {type(e).__name__}: {e}")
                        break

                if replay_ok:
                    for idx, (logged_ascii, logged_action) in enumerate(parsed_steps):
                        current_ascii = renderer.render(backend, level, objs)
                        if current_ascii.strip() != logged_ascii.strip():
                            replay_ok = False
                            print(f"Resume verification failed at step {idx+1}: current ASCII does not match step log input state.")
                            break

                        state_history_lst.append((current_ascii, logged_action))
                        result["action_sequence"].append(int(logged_action))
                        action_sequence_for_render.append(int(logged_action))

                        prev_score = score
                        _, _, _, _, score, level, _, objs = backend.solver.takeAction(backend.engine, int(logged_action))
                        won = bool(backend.engine.getWinning())
                        reward = float(score) - float(prev_score)
                        if won:
                            reward += 1.0
                        reward -= 0.01
                        next_ascii = renderer.render(backend, level, list(objs))

                        result["reward_sequence"].append(float(reward))
                        result["heuristic_sequence"].append(float(score))
                        result["state_data"] = {
                            "score": float(score),
                            "win": won,
                            "step": idx + 1,
                        }

                        if idx + 1 < len(parsed_steps):
                            expected_next_ascii, _ = parsed_steps[idx + 1]
                            if next_ascii.strip() != expected_next_ascii.strip():
                                replay_ok = False
                                print(f"Resume verification failed between steps {idx+1} and {idx+2}: next state does not match next step log input state.")
                                break

                        replayed_steps += 1
                        ascii_state = next_ascii
                        objs = list(objs)
                        if won:
                            result["win"] = True
                            print(f"Replay reached terminal win at step {idx+1}.")
                            break

                if replayed_steps > 0:
                    print(f"Resumed {replayed_steps} step(s) from logs.")
                elif step_files:
                    print("Resume replay was not applied; continuing from fresh initial state.")

        for step in range(replayed_steps, max_steps):
            print(f"\nStep {step+1}/{max_steps}")
            log_file = os.path.join(current_run_logs_dir, f"step_{step+1}.txt")

            action_id = agent.choose_action(
                ascii_map=ascii_state,
                rules=rules,
                action_space=ACTION_SPACE,
                action_meanings=ACTION_MEANINGS,
                think_aloud=think_aloud,
                memory=memory,
                state_history=state_history_lst,
                log_file=log_file,
            )

            state_history_lst.append((ascii_state, action_id))
            action_str = ACTION_MEANINGS[action_id]
            print(f"LLM chose action id: {action_id} ({action_str})")
            result["action_sequence"].append(int(action_id))
            action_sequence_for_render.append(int(action_id))

            prev_score = score
            _, _, _, _, score, level, _, objs = backend.solver.takeAction(backend.engine, int(action_id))
            won = bool(backend.engine.getWinning())
            reward = float(score) - float(prev_score)
            if won:
                reward += 1.0
            reward -= 0.01
            objs = list(objs)
            ascii_state = renderer.render(backend, level, objs)

            print("New state (ASCII):")
            print(ascii_state)
            print(f"Reward: {reward} | Win: {won}")

            result["reward_sequence"].append(float(reward))
            result["heuristic_sequence"].append(float(score))
            result["state_data"] = {
                "score": float(score),
                "win": won,
                "step": step + 1,
            }

            if won:
                print(f"Game completed in {step+1} steps! (Win)")
                result["win"] = True
                break

        result["heuristics"] = [float(h) for h in result["heuristic_sequence"]]
        del result["heuristic_sequence"]
        result["final_ascii"] = ascii_state.split("\n")
        if getattr(process_game_level, "_render_default", True):
            try:
                gif_path = current_run_filepath[:-5] + ".gif"
                backend.render_gif(
                    game_text=game_info["game_text"],
                    level_i=level_index,
                    actions=action_sequence_for_render,
                    gif_path=gif_path,
                    frame_duration_s=0.1,
                    scale=10,
                )
                result["render_gif"] = gif_path
                print(f"Episode GIF saved to {gif_path}")
            except Exception as e:
                print(f"Warning: failed to render episode GIF for {current_run_filepath}: {type(e).__name__}: {e}")

        with open(current_run_filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {current_run_filepath} (Run ID: {run_id})")
        return True

    except Exception as e:
        print(f"!!! ERROR during Game: {game_name}, Level: {level_index}, Run: {run_id} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        return False


def worker_loop(args, all_jobs, game_info_map, save_dir_main):
    worker_id = args.worker_id if args.worker_id else f"{os.uname().nodename}-{os.getpid()}"
    state_path = args.state_path if args.state_path else os.path.join(save_dir_main, STATE_FILE_BASENAME)
    print(f"Worker {worker_id}: starting with state_path {state_path}")

    while True:
        job = claim_next_job(
            state_path=state_path,
            all_jobs=all_jobs,
            save_dir=save_dir_main,
            model=args.model,
            think_aloud=args.think_aloud,
            memory=args.memory,
            force=args.force,
            worker_id=worker_id,
        )
        if job is None:
            print(f"Worker {worker_id}: no remaining claimable jobs. Exiting worker loop.")
            break

        game_name, level_index, run_id = job
        print(f"Worker {worker_id} processing: {game_name} | Level {level_index} | Run {run_id}")
        enable_thinking = args.enable_thinking if args.model.startswith("vllm") else None
        agent = LLMGameAgent(model_name=args.model, enable_thinking=enable_thinking)
        success = process_game_level(
            agent=agent,
            game_info=game_info_map[game_name],
            level_index=level_index,
            run_id=run_id,
            save_dir=save_dir_main,
            think_aloud=args.think_aloud,
            model=args.model,
            max_steps=args.max_steps,
            memory=args.memory,
            force=args.force,
        )
        release_job(state_path, args.model, game_name, level_index, run_id, args.think_aloud, args.memory, "done" if success else "failed")


def main():
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="LLM agent loop experiment using puzzlescript_nodejs")
    parser.add_argument('--model', type=str, required=True,
                        choices=['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen',
                                 'deepseek-r1', 'llama',
                                 'vllm', 'vllm-qwen3', 'vllm-qwen3-4b', 'vllm-qwen3-30b', 'vllm-qwen3-32b',
                                 'vllm-llama3', 'vllm-llama3-70b', 'vllm-mistral',
                                 'vllm-deepseek', 'vllm-deepseek-r1'])
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--resume_game_name', type=str, default='')
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--run_id_start', type=int, default=1)
    parser.add_argument('--think_aloud', action='store_true')
    parser.add_argument('--memory', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="llm_agent_results_nodejs")
    parser.add_argument('--game', type=str, default='')
    parser.add_argument('--worker_id', type=str, default='')
    parser.add_argument('--state_path', type=str, default='')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--inflight_jobs', type=int, default=1)
    parser.add_argument('--vllm_base_url', type=str, default='')
    parser.add_argument('--enable_thinking', action='store_true', default=False)
    parser.add_argument('--no_render', action='store_true')
    args = parser.parse_args()
    process_game_level._render_default = not args.no_render

    if args.vllm_base_url:
        os.environ['VLLM_BASE_URL'] = args.vllm_base_url

    if args.game:
        game_names = [args.game]
        print(f"Running specific game: {args.game}")
    else:
        game_names = PRIORITY_GAMES.copy()
        resume_game_name = args.resume_game_name
        actual_resume_game_name = 'atlas shrank' if resume_game_name == 'atlas_shrank' else resume_game_name
        if actual_resume_game_name:
            try:
                start_idx = game_names.index(actual_resume_game_name)
                game_names = game_names[start_idx:]
            except ValueError:
                print(f"Warning: Game '{actual_resume_game_name}' not found in PRIORITY_GAMES list. Starting from the beginning.")

    if args.reverse:
        game_names.reverse()
        print(f"Processing games in reverse order, starting with: {game_names[0]}")
    else:
        print(f"Processing games in order, starting with: {game_names[0]}")

    print(f"\n=== Running NodeJS LLM agent with model: {args.model}, for {len(game_names)} games, ensuring up to {args.num_runs} total runs per level ===")

    model_folder = args.model if args.memory <= 0 else f"{args.model}_mem-{args.memory}"
    save_dir_main = os.path.join(args.save_dir, model_folder)
    os.makedirs(save_dir_main, exist_ok=True)

    start_level = args.level
    print(f"Starting from level {start_level}")

    print("\n==== Collecting information for all games ====")
    all_games_info = []
    for game_idx, game_name in enumerate(game_names):
        print(f"\n-- Checking game {game_idx+1}/{len(game_names)}: {game_name} --")
        game_info = collect_game_info(game_name, start_level)
        if game_info:
            all_games_info.append(game_info)
            print(f"Added game {game_name} with {game_info['num_levels']} levels, will process levels {game_info['levels_to_process']}")

    if not all_games_info:
        print("No valid games found. Exiting.")
        return

    all_jobs = []
    for run_id in range(args.run_id_start, args.num_runs + args.run_id_start):
        for game_info in all_games_info:
            for level_index in game_info["levels_to_process"]:
                all_jobs.append((game_info["game_name"], level_index, run_id))

    game_info_map = {gi["game_name"]: gi for gi in all_games_info}

    if args.workers > 1:
        processes = []
        for _ in range(args.workers):
            p = multiprocessing.Process(target=worker_loop, args=(args, all_jobs, game_info_map, save_dir_main))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        worker_loop(args, all_jobs, game_info_map, save_dir_main)

    print("\n=== All runs for all games and levels have been processed ===")


if __name__ == "__main__":
    main()
