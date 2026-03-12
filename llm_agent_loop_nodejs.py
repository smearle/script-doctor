import argparse
import multiprocessing
import os
import json
from dataclasses import dataclass, field

from lark import Lark

from LLM_agent import LLMGameAgent
from ascii_prompting import ASCIIStateFormatter
from llm_agent_loop import (
    CUSTOM_GAMES_DIR,
    STATE_FILE_BASENAME,
    claim_next_job,
    get_run_file_path,
    get_run_logs_dir,
    parse_legend,
    release_job,
)
from backends.nodejs import NodeJSPuzzleScriptBackend
from puzzlescript_jax.globals import LARK_SYNTAX_PATH, PRIORITY_GAMES
from puzzlescript_jax.utils import level_to_int_arr


ACTION_SPACE = [0, 1, 2, 3, 4, 5, 6]
ACTION_MEANINGS = {0: "left", 1: "down", 2: "right", 3: "up", 4: "action", 5: "undo", 6: "restart"}


@dataclass
class NodeJSAsciiRenderer:
    legend_mapping: dict[str, list[str]]
    formatter: ASCIIStateFormatter = field(init=False)

    def __post_init__(self):
        self.formatter = ASCIIStateFormatter(self.legend_mapping)

    def _build_name_grid(self, backend: NodeJSPuzzleScriptBackend, level, objs: list[str]):
        level_arr = level_to_int_arr(level, len(objs))
        mini, minj, maxi, maxj = backend._get_visible_bounds(level)
        name_grid = []
        for y in range(minj, maxj):
            row = []
            for x in range(mini, maxi):
                cell_bits = int(level_arr[x, y])
                names = [objs[obj_i] for obj_i in range(len(objs)) if ((cell_bits >> obj_i) & 1) == 1]
                row.append(tuple(names))
            name_grid.append(row)
        return name_grid

    def render(self, backend: NodeJSPuzzleScriptBackend, level, objs: list[str]) -> str:
        """Return LEGEND + MAP combined string (backward compat)."""
        return self.formatter.render_from_name_grid(self._build_name_grid(backend, level, objs))

    def render_map(self, backend: NodeJSPuzzleScriptBackend, level, objs: list[str]) -> str:
        """Return only the map string (no legend header). Registers new combos."""
        return self.formatter.render_map_only(self._build_name_grid(backend, level, objs))

    def get_legend_text(self) -> str:
        """Return the full legend for all combos encountered so far."""
        return self.formatter.get_legend_text()


def collect_game_info(game_name):
    game_path = os.path.join(CUSTOM_GAMES_DIR, f"{game_name}.txt")
    if not os.path.exists(game_path):
        print(f"Error: Game file not found at {game_path}. Skipping game.")
        return None

    legend_lines = _extract_section(game_path, "LEGEND")
    legend_mapping = parse_legend(legend_lines)

    try:
        with open(LARK_SYNTAX_PATH, "r", encoding="utf-8") as f:
            puzzlescript_grammar = f.read()
        grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
        backend = NodeJSPuzzleScriptBackend()
        game_text = backend.compile_game(grammar_parser, game_name)
        num_levels = backend.get_num_levels()

        metadata = backend.engine.getState().metadata
        title = str(getattr(metadata, "title", game_name))
        author = str(getattr(metadata, "author", ""))

        level_info = list(backend.engine.getLevelInfo())
        backend.unload_game()
    except Exception as e:
        print(f"Error collecting NodeJS game info for {game_name}: {type(e).__name__}, {e}")
        return None

    if num_levels <= 0:
        print(f"Error: No levels found for game '{game_name}'. Skipping game.")
        return None

    return {
        "game_name": game_name,
        "game_path": game_path,
        "game_text": game_text,
        "legend_mapping": legend_mapping,
        "title": title,
        "author": author,
        "num_levels": num_levels,
        "level_info": level_info,
    }


def _extract_section(filepath, section):
    """Extract a section from a PuzzleScript game file."""
    import re as _re
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    section_lines = []
    in_section = False
    for line in lines:
        if not in_section and _re.match(r"=*\s*%s\s*=*" % section, line.strip(), _re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            if _re.match(r"=+\s*[A-Z_]+\s*=+", line.strip()) or (line.strip().isupper() and len(line.strip()) > 3):
                break
            section_lines.append(line.rstrip())
    while section_lines and not section_lines[0].strip():
        section_lines.pop(0)
    while section_lines and not section_lines[-1].strip():
        section_lines.pop()
    return section_lines


def process_full_game(agent, game_info, run_id, save_dir, model,
                      max_steps_per_level, history_limit, no_render=False, force=False):
    """Play through the entire game: messages are shown, playable levels are played sequentially."""
    game_name = game_info["game_name"]
    game_text = game_info["game_text"]
    title = game_info["title"]
    author = game_info["author"]
    level_info_list = game_info["level_info"]
    num_levels = game_info["num_levels"]

    formatted_game_name = game_name.replace(" ", "_")
    run_filepath = os.path.join(save_dir, f"{formatted_game_name}_run_{run_id}.json")
    logs_dir = run_filepath[:-5] + "_logs"
    os.makedirs(logs_dir, exist_ok=True)

    if os.path.exists(run_filepath) and not force:
        print(f"Run {run_id} for Game: {game_name} already exists. Skipping.")
        return True

    try:
        with open(run_filepath, "w", encoding="utf-8") as f:
            f.write('{"status": "running"}')
    except Exception as e:
        print(f"Failed to create lock file for {run_filepath}: {e}")
        return False

    try:
        backend = NodeJSPuzzleScriptBackend()
        renderer = NodeJSAsciiRenderer(game_info["legend_mapping"])

        scratchpad = ""
        state_history = []  # list of (map_str, action_id) across the whole game
        pending_messages = []
        playable_level_number = 0

        result = {
            "model": model,
            "game": game_name,
            "title": title,
            "author": author,
            "run": run_id,
            "levels": [],
            "game_complete": False,
        }

        for lvl_idx in range(num_levels):
            lv_info = level_info_list[lvl_idx]
            lv_type = str(lv_info["type"])

            if lv_type == "message":
                msg_text = str(lv_info.get("message", ""))
                pending_messages.append(msg_text)
                result["levels"].append({
                    "level_index": lvl_idx,
                    "type": "message",
                    "message": msg_text,
                })
                print(f"\n--- Message (level index {lvl_idx}): {msg_text} ---")
                continue

            # Playable level
            playable_level_number += 1
            print(f"\n=== Game: {game_name}, Playable level {playable_level_number} "
                  f"(index {lvl_idx}), Run: {run_id} ===")

            backend.load_level(game_text, lvl_idx)
            backend.solver.precalcDistances(backend.engine)

            level = backend.engine.backupLevel()
            objs = list(backend.engine.getState().idDict)
            score = float(backend.solver.getScore(backend.engine))
            won = bool(backend.engine.getWinning())
            ascii_map = renderer.render_map(backend, level, objs)
            legend_text = renderer.get_legend_text()

            level_result = {
                "level_index": lvl_idx,
                "playable_level_number": playable_level_number,
                "type": "level",
                "win": False,
                "messages_before": list(pending_messages),
                "initial_ascii": ascii_map.split("\n"),
                "action_sequence": [],
                "heuristic_sequence": [],
                "scratchpad_sequence": [],
            }
            action_sequence_for_render = []

            messages_for_prompt = list(pending_messages) if pending_messages else None
            pending_messages.clear()

            for step in range(max_steps_per_level):
                if won:
                    break

                print(f"\nLevel {playable_level_number}, Step {step+1}/{max_steps_per_level}")
                log_file = os.path.join(
                    logs_dir, f"level_{lvl_idx}_step_{step+1}.txt"
                )

                action_id, scratchpad = agent.choose_action_human(
                    title=title,
                    author=author,
                    legend_text=legend_text,
                    ascii_map=ascii_map,
                    action_space=ACTION_SPACE,
                    action_meanings=ACTION_MEANINGS,
                    state_history=state_history,
                    history_limit=history_limit,
                    scratchpad=scratchpad,
                    messages=messages_for_prompt,
                    level_number=playable_level_number,
                    log_file=log_file,
                )

                # Only show messages on the first step of a new level
                messages_for_prompt = None

                state_history.append((ascii_map, action_id))

                action_str = ACTION_MEANINGS[action_id]
                print(f"LLM chose action id: {action_id} ({action_str})")
                level_result["action_sequence"].append(int(action_id))
                level_result["scratchpad_sequence"].append(scratchpad)
                action_sequence_for_render.append(int(action_id))

                prev_score = score
                _, _, _, _, score, level, _, objs = backend.solver.takeAction(
                    backend.engine, int(action_id)
                )
                won = bool(backend.engine.getWinning())
                objs = list(objs)
                ascii_map = renderer.render_map(backend, level, objs)
                legend_text = renderer.get_legend_text()

                level_result["heuristic_sequence"].append(float(score))
                print(f"Score: {score} | Win: {won}")

                if won:
                    print(f"Level {playable_level_number} completed in {step+1} steps!")
                    level_result["win"] = True
                    break

            level_result["final_ascii"] = ascii_map.split("\n")
            level_result["final_scratchpad"] = scratchpad
            level_result["steps_taken"] = len(level_result["action_sequence"])

            # Render GIF for this level
            if not no_render:
                try:
                    gif_path = os.path.join(
                        logs_dir,
                        f"level_{lvl_idx}.gif",
                    )
                    backend.render_gif(
                        game_text=game_text,
                        level_i=lvl_idx,
                        actions=action_sequence_for_render,
                        gif_path=gif_path,
                        frame_duration_s=0.05,
                        scale=10,
                    )
                    level_result["render_gif"] = gif_path
                    print(f"Level GIF saved to {gif_path}")
                except Exception as e:
                    print(f"Warning: failed to render GIF for level {lvl_idx}: "
                          f"{type(e).__name__}: {e}")

            result["levels"].append(level_result)

            if not won:
                print(f"Level {playable_level_number} not completed within "
                      f"{max_steps_per_level} steps. Stopping game.")
                break

        else:
            # All levels completed (loop didn't break)
            result["game_complete"] = True
            print(f"\n=== Game '{game_name}' completed! ===")

        with open(run_filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {run_filepath} (Run ID: {run_id})")
        return True

    except Exception as e:
        print(f"!!! ERROR during Game: {game_name}, Run: {run_id} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
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
            think_aloud=False,
            memory=args.history_limit,
            force=args.force,
            worker_id=worker_id,
        )
        if job is None:
            print(f"Worker {worker_id}: no remaining claimable jobs. Exiting worker loop.")
            break

        game_name, _level_index, run_id = job
        print(f"Worker {worker_id} processing: {game_name} | Run {run_id}")
        enable_thinking = args.enable_thinking if args.model.startswith("vllm") else None
        agent = LLMGameAgent(model_name=args.model, enable_thinking=enable_thinking)
        success = process_full_game(
            agent=agent,
            game_info=game_info_map[game_name],
            run_id=run_id,
            save_dir=save_dir_main,
            model=args.model,
            max_steps_per_level=args.max_steps,
            history_limit=args.history_limit,
            no_render=args.no_render,
            force=args.force,
        )
        release_job(state_path, args.model, game_name, _level_index, run_id, False, args.history_limit, "done" if success else "failed")


def main():
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Human-like LLM agent loop (plays full games) using puzzlescript_nodejs")
    parser.add_argument('--model', type=str, required=True,
                        choices=['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen',
                                 'deepseek-r1', 'llama',
                                 'vllm', 'vllm-qwen3', 'vllm-qwen3-4b', 'vllm-qwen3-30b', 'vllm-qwen3-32b',
                                 'vllm-llama3', 'vllm-llama3-70b', 'vllm-mistral',
                                 'vllm-deepseek', 'vllm-deepseek-r1', 'vllm-qwen3.5-27b-fp8', 'vllm-qwen3.5-9b'])
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps per playable level')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--resume_game_name', type=str, default='')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--run_id_start', type=int, default=1)
    parser.add_argument('--history_limit', type=int, default=10,
                        help='Number of recent (map, action) pairs shown to agent')
    parser.add_argument('--save_dir', type=str, default="llm_agent_results_nodejs")
    parser.add_argument('--game', type=str, default='')
    parser.add_argument('--worker_id', type=str, default='')
    parser.add_argument('--state_path', type=str, default='')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--vllm_base_url', type=str, default='')
    parser.add_argument('--enable_thinking', action='store_true', default=False)
    parser.add_argument('--no_render', action='store_true')
    args = parser.parse_args()

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

    print(f"\n=== Running human-like NodeJS LLM agent with model: {args.model}, "
          f"for {len(game_names)} games, up to {args.num_runs} runs each ===")

    save_dir_main = os.path.join(args.save_dir, args.model)
    os.makedirs(save_dir_main, exist_ok=True)

    print("\n==== Collecting information for all games ====")
    all_games_info = []
    for game_idx, game_name in enumerate(game_names):
        print(f"\n-- Checking game {game_idx+1}/{len(game_names)}: {game_name} --")
        game_info = collect_game_info(game_name)
        if game_info:
            all_games_info.append(game_info)
            print(f"Added game '{game_name}' ({game_info['title']}) "
                  f"with {game_info['num_levels']} levels")

    if not all_games_info:
        print("No valid games found. Exiting.")
        return

    # Jobs are (game_name, 0, run_id) — level_index 0 is a placeholder since we play full games
    all_jobs = []
    for run_id in range(args.run_id_start, args.num_runs + args.run_id_start):
        for game_info in all_games_info:
            all_jobs.append((game_info["game_name"], 0, run_id))

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

    print("\n=== All runs for all games have been processed ===")


if __name__ == "__main__":
    main()
