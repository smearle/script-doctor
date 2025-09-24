import argparse
import os
import re
import json
import time
import fcntl
import multiprocessing
import jax
from lark import Lark
from env_wrappers import RepresentationWrapper
from env import PSParams
from preprocess_games import PS_LARK_GRAMMAR_PATH, get_tree_from_txt
from LLM_agent import LLMGameAgent
from globals import PRIORITY_GAMES


CUSTOM_GAMES_DIR = "data/scraped_games"

# Shared state file for coordinating parallel workers
STATE_FILE_BASENAME = "work_state.json"


def make_job_key(model, game_name, level_index, run_id, think_aloud, memory):
    cot = "CoT" if think_aloud else "NoCoT"
    mem = f"mem-{memory}" if memory and memory > 0 else "mem-0"
    return f"{model}|{mem}|{cot}|{game_name}|{level_index}|{run_id}"


def claim_next_job(state_path, all_jobs, save_dir, model, think_aloud, memory, force, worker_id):
    """
    Atomically claim the next available job by updating the shared JSON state.
    Returns a tuple (game_name, level_index, run_id) or None if no jobs remain.
    """
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            state = json.load(f)
        except Exception:
            state = {}
        working = state.get("working", {})
        completed = set(state.get("completed", []))

        # Mark already-existing outputs as completed when not forcing
        if not force:
            for game_name, level_index, run_id in all_jobs:
                key = make_job_key(model, game_name, level_index, run_id, think_aloud, memory)
                if key in completed or key in working:
                    continue
                if check_run_file_exists(save_dir, model, game_name, run_id, level_index, think_aloud, memory):
                    completed.add(key)

        chosen = None
        for game_name, level_index, run_id in all_jobs:
            key = make_job_key(model, game_name, level_index, run_id, think_aloud, memory)
            if key in completed or key in working:
                continue
            if (not force) and check_run_file_exists(save_dir, model, game_name, run_id, level_index, think_aloud, memory):
                completed.add(key)
                continue
            # Claim this job
            working[key] = {"worker": worker_id, "ts": time.time()}
            chosen = (game_name, level_index, run_id)
            break

        state["working"] = working
        state["completed"] = sorted(completed)
        f.seek(0)
        json.dump(state, f, indent=2)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)

    return chosen


def release_job(state_path, model, game_name, level_index, run_id, think_aloud, memory, status):
    """
    Release a claimed job from the working set and optionally mark completed.
    Status can be 'done', 'failed', or 'skipped'.
    """
    key = make_job_key(model, game_name, level_index, run_id, think_aloud, memory)
    with open(state_path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            state = json.load(f)
        except Exception:
            state = {}
        working = state.get("working", {})
        completed = set(state.get("completed", []))

        if key in working:
            del working[key]
        if status in ("done", "skipped"):
            completed.add(key)

        state["working"] = working
        state["completed"] = sorted(completed)
        f.seek(0)
        json.dump(state, f, indent=2)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)


def worker_loop(args, all_jobs, game_info_map, save_dir_main):
    worker_id = args.worker_id if args.worker_id else f"{os.uname().nodename}-{os.getpid()}"
    state_path = args.state_path if args.state_path else os.path.join(save_dir_main, STATE_FILE_BASENAME)
    # Initialize a new LLM agent for this worker
    agent = LLMGameAgent(model_name=args.model)
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
        game_info = game_info_map[game_name]
        success = process_game_level(
            agent=agent,
            game_info=game_info,
            level_index=level_index,
            run_id=run_id,
            save_dir=save_dir_main,
            think_aloud=args.think_aloud,
            model=args.model,
            max_steps=args.max_steps,
            memory=args.memory,
            force=args.force
        )
        release_job(state_path, args.model, game_name, level_index, run_id, args.think_aloud, args.memory, 'done' if success else 'failed')


def extract_section(filepath, section):
    """
    Extract a specific section from a PuzzleScript game file.
    
    Args:
        filepath: Path to the game file
        section: Section name to extract (e.g., "RULES", "LEGEND", "LEVELS")
        
    Returns:
        List of lines in the specified section
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    section_lines = []
    in_section = False
    for line in lines:
        if not in_section and re.match(r"=*\s*%s\s*=*" % section, line.strip(), re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            if re.match(r"=+\s*[A-Z_]+\s*=+", line.strip()) or (line.strip().isupper() and len(line.strip()) > 3):
                break
            section_lines.append(line.rstrip())
    while section_lines and not section_lines[0].strip():
        section_lines.pop(0)
    while section_lines and not section_lines[-1].strip():
        section_lines.pop()
    return section_lines


def parse_legend(legend_lines):
    """
    Parse the LEGEND section from a PuzzleScript game file.
    
    Args:
        legend_lines: Lines from the LEGEND section
        
    Returns:
        Dictionary mapping legend keys to objects
    """
    mapping = {}
    for line in legend_lines:
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            objs = [obj.strip() for obj in v.strip().split()]
            mapping[k] = objs
    return mapping


def extract_first_level(level_lines):
    """
    Extract the first level from the LEVELS section.
    
    Args:
        level_lines: Lines from the LEVELS section
        
    Returns:
        String representation of the first level
    """
    levels = []
    current = []
    for line in level_lines:
        if not line.strip():
            if current:
                levels.append(current)
                current = []
        else:
            current.append(line)
    if current:
        levels.append(current)
    if levels:
        return "\n".join(levels[0])
    return ""


def check_run_file_exists(save_dir, model, game_name, run_id, level_index, think_aloud, memory):
    """
    Check if the specified run file exists, supporting both new (folder-based) and legacy naming.
    
    New style (this change): files are stored under save_dir (which is model[_mem] folder),
    and filename does NOT contain model/memory prefix:
      [CoT_]{game}_run_{run}_level_{level}.json
      [CoT_]{game}_run_{run}.json  (for level 0)
    
    Legacy style (back-compat, still checked in the same folder):
      {model}_[mem-#_]{CoT_}{game}_run_{run}_level_{level}.json
      {model}_{CoT_}{game}_run_{run}.json (sometimes without mem-# for level 0)
    """
    # Format the game name by replacing spaces with underscores for file paths
    formatted_game_name = game_name.replace(" ", "_")
    cot_prefix = "CoT_" if think_aloud else ""
    memory_prefix = f"mem-{str(memory)}_" if memory and memory > 0 else ""

    # New style: inside save_dir (model[_mem] folder), filenames don't include model/memory
    new_filename_with_level = f"{cot_prefix}{formatted_game_name}_run_{run_id}_level_{level_index}.json"
    new_path_with_level = os.path.join(save_dir, new_filename_with_level)

    new_filename_without_level = f"{cot_prefix}{formatted_game_name}_run_{run_id}.json"
    new_path_without_level = os.path.join(save_dir, new_filename_without_level)

    # Legacy style: same folder, model/memory prefixes in filename
    legacy_filename_with_level = f"{model}_{memory_prefix}{cot_prefix}{formatted_game_name}_run_{run_id}_level_{level_index}.json"
    legacy_path_with_level = os.path.join(save_dir, legacy_filename_with_level)

    legacy_orig_filename_with_level = f"{model}_{memory_prefix}{cot_prefix}{game_name}_run_{run_id}_level_{level_index}.json"
    legacy_orig_path_with_level = os.path.join(save_dir, legacy_orig_filename_with_level)

    # Legacy level-0 variants (some runs didn't include memory prefix on level-0 file)
    legacy_filename_without_level = f"{model}_{cot_prefix}{formatted_game_name}_run_{run_id}.json"
    legacy_path_without_level = os.path.join(save_dir, legacy_filename_without_level)

    legacy_orig_filename_without_level = f"{model}_{cot_prefix}{game_name}_run_{run_id}.json"
    legacy_orig_path_without_level = os.path.join(save_dir, legacy_orig_filename_without_level)

    if level_index == 0:
        exists = (
            os.path.exists(new_path_with_level) or
            os.path.exists(new_path_without_level) or
            os.path.exists(legacy_path_with_level) or
            os.path.exists(legacy_orig_path_with_level) or
            os.path.exists(legacy_path_without_level) or
            os.path.exists(legacy_orig_path_without_level)
        )
        return exists

    exists = os.path.exists(new_path_with_level) or os.path.exists(legacy_path_with_level) or os.path.exists(legacy_orig_path_with_level)
    return exists


def get_run_file_path(save_dir, model, game_name, run_id, level_index, think_aloud: bool, memory: int):
    """
    Get file path for saving run results using folder-based model/memory separation.
    save_dir is expected to be the model[_mem] folder already.
    Filenames will not include model/memory prefix.
    """
    formatted_game_name = game_name.replace(" ", "_")
    filename = (('CoT_' if think_aloud else '') +
                f"{formatted_game_name}_run_{run_id}_level_{level_index}.json")
    return os.path.join(save_dir, filename)


def find_next_available_run_id(save_dir, model, game_name, level_index, initial_run_id, think_aloud, memory):
    """
    Find the next available run ID that doesn't conflict with existing files.
    
    Args:
        save_dir: Directory where run results are saved (model[_mem] folder)
        model: Model name
        game_name: Game name
        level_index: Level index
        initial_run_id: The initial run ID we'd like to use
        think_aloud: Whether to use CoT prefix in filename
        
    Returns:
        int: The next available run ID
    """
    run_id = initial_run_id
    while check_run_file_exists(save_dir, model, game_name, run_id, level_index, think_aloud, memory):
        print(f"Run {run_id} already exists for Game: {game_name}, Level: {level_index}, CoT: {think_aloud}, Memory: {str(memory)}, Trying next run ID.")
        run_id += 1
    return run_id


def collect_game_info(game_name, start_level):
    """
    Collect information about a specific game, including rules, levels, etc.
    
    Args:
        game_name: Name of the game
        start_level: Starting level index
        
    Returns:
        Dictionary containing game information, or None if there was an error
    """
    game_path = os.path.join(CUSTOM_GAMES_DIR, f"{game_name}.txt")
    if not os.path.exists(game_path):
        print(f"Error: Game file not found at {game_path}. Skipping game.")
        return None

    # Extract sections for LLM agent
    rules_lines = extract_section(game_path, "RULES")
    legend_lines = extract_section(game_path, "LEGEND")
    level_lines = extract_section(game_path, "LEVELS")
    rules = "\n".join(rules_lines)
    mapping = parse_legend(legend_lines)
    ascii_map = extract_first_level(level_lines)

    if not rules or not mapping or not ascii_map:
        print(f"Error: Missing rules, mapping, or initial level ASCII for game {game_name}. Skipping game.")
        return None

    # Initialize environment parser
    try:
        with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
            puzzlescript_grammar = f.read()
        grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
        tree, success, err_msg = get_tree_from_txt(grammar_parser, game_name, test_env_init=False)
        if success != 0:
            print(f"Error: Failed to parse game file for {game_name}. Skipping game.")
            print(f"Parse error: {err_msg}, code: {success}")
            return None

        # Do NOT create env here! Only store info needed to reconstruct it in the worker.
        # Instead, check levels by parsing tree directly.
        levels = getattr(tree, "levels", None)
        if not levels or len(levels) == 0:
            print(f"Error: No levels found for game '{game_name}'. Skipping game.")
            return None

        # Collect game information (no env object)
        game_info = {
            "game_name": game_name,
            "game_path": game_path,
            "rules": rules,
            "mapping": mapping,
            "ascii_map": ascii_map,
            "tree": tree,
            "num_levels": len(levels),
            "levels_to_process": list(range(start_level, len(levels)))
        }
        return game_info

    except Exception as e:
        print(f"Error collecting game info for {game_name}: {type(e).__name__}, {e}")
        return None


def process_game_level(agent, game_info, level_index, run_id, save_dir, model,
                       max_steps, think_aloud: bool, memory: int, force=False):
    """
    Process a specific game level for a specific run ID.
    
    Args:
        agent: LLM agent
        game_info: Dictionary containing game information
        level_index: Level index to process
        run_id: Run ID
        save_dir: Directory to save results (model[_mem] folder)
        model: Model name
        max_steps: Maximum steps per episode
        force: Whether to force rerun existing results
        
    Returns:
        Boolean indicating success
    """
    game_name = game_info["game_name"]
    # Reconstruct env inside the worker process
    from env_wrappers import RepresentationWrapper
    tree = game_info["tree"]
    env = RepresentationWrapper(tree, debug=False, print_score=False)
    rules = game_info["rules"]
    
    print(f"\n=== Processing Game: {game_name}, Level: {level_index}, Run: {run_id} ===")
    
    # Get the path for saving results
    current_run_filepath = get_run_file_path(save_dir, model, game_name, run_id,
                                             level_index, think_aloud, memory)

    # Create an empty run file as a lock before starting work
    if os.path.exists(current_run_filepath) and not force:
        print(f"Run {run_id} for Game: {game_name}, Level: {level_index} already exists. Skipping.")
        return True
    try:
        with open(current_run_filepath, "w", encoding="utf-8") as f:
            json.dump({"status": "running"}, f)
    except Exception as e:
        print(f"Failed to create lock file for {current_run_filepath}: {e}")
        return False

    current_run_logs_dir = current_run_filepath[:-5] + "_logs"
    os.makedirs(current_run_logs_dir, exist_ok=True)
    
    try:
        # Ensure level_index is valid
        if level_index < 0 or level_index >= game_info["num_levels"]:
            print(f"Warning: Invalid level_index {level_index} for game '{game_name}' (max index {game_info['num_levels']-1}). Skipping level.")
            return False
        
        level = env.get_level(level_index)
        env_params = PSParams(level=level)
        # Use run_id as seed to ensure uniqueness
        rng = jax.random.PRNGKey(run_id * 1000 + level_index)
        
        obs, state = env.reset(rng=rng, params=env_params)
        ascii_state = env.render_ascii_and_legend(state)
        
        action_space = [0, 1, 2, 3, 4]
        action_meanings = {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}
        
        result = {
            "model": model,
            "game": game_name,
            "level": level_index,
            "run": run_id,
            "win": False,
            "action_sequence": [],
            "reward_sequence": [],
            "heuristic_sequence": []
        }
        
        state_history = set()
        state_history_lst = []
        current_state = state
        
        # Main action loop
        for step in range(max_steps):
            print(f"\nStep {step+1}/{max_steps}")
            
            h = hash(current_state.multihot_level.tobytes())
            state_history.add(h)

            log_file = os.path.join(current_run_logs_dir, f"step_{step+1}.txt")
            
            action_id = agent.choose_action(
                ascii_map=ascii_state,
                rules=rules,
                action_space=action_space,
                action_meanings=action_meanings,
                think_aloud=think_aloud,
                memory=memory,
                state_history=state_history_lst,
                log_file=log_file,
            )

            state_history_lst.append((ascii_state, action_id))
            
            action_str = action_meanings[action_id]
            print(f"LLM chose action id: {action_id} ({action_str})")
            result["action_sequence"].append(int(action_id))
            
            rng, _rng = jax.random.split(rng)
            obs, next_state, rew, done, info = env.step_env(
                rng=rng, action=action_id, state=current_state, params=env_params
            )
            ascii_state = env.render_ascii_and_legend(next_state)
            print("New state (ASCII):")
            print(ascii_state)
            print(f"Reward: {rew} | Win: {next_state.win}")
            
            result["reward_sequence"].append(float(rew.item()))
            
            if step == 0:
                result["initial_ascii"] = ascii_state.split('\n')
            
            result["heuristic_sequence"].append(float(next_state.heuristic.item()))
            
            result["state_data"] = {
                "score": int(next_state.score),
                "win": bool(next_state.win),
                "step": step + 1
            }
            
            if next_state.win:
                print(f"Game completed in {step+1} steps! (Win)")
                result["win"] = True
                break
            if done:
                print(f"Game ended in {step+1} steps (done=True, not win).")
                break
            
            current_state = next_state
        
        # Process final results
        result["heuristics"] = [float(h) for h in result["heuristic_sequence"]]
        if "heuristic_sequence" in result:
            del result["heuristic_sequence"]
        result["final_ascii"] = ascii_state.split('\n')
        
        # Save results to the originally-claimed run file path (do not change run_id under parallel workers)
        if os.path.exists(current_run_filepath):
            print(f"Result file already exists at {current_run_filepath}. Not overwriting.")
        else:
            with open(current_run_filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=lambda o: f"<{type(o).__name__} instance>")
            print(f"Result saved to {current_run_filepath} (Run ID: {run_id})")
        
        return True
        
    except Exception as e:
        print(f"!!! ERROR during Game: {game_name}, Level: {level_index}, Run: {run_id} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        return False


def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description='LLM agent loop experiment (env+rules/ascii/mapping)')
    parser.add_argument('--model', type=str, required=True, choices=['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen', 'deepseek-r1', 'llama'],
                        help='LLM model alias (4o-mini=4o-mini, o3=O3-mini, gemini=Gemini-2.0, gemini-2.5-Pro, deepseek=DeepSeek, qwen=Qwen, llama=Llama-3 via Portkey)')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode (default: 100)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs per game (default: 10)')
    parser.add_argument('--reverse', action='store_true',
                        help='Process games in reverse order')
    parser.add_argument('--resume_game_name', type=str, default='',
                        help='Name of the game to resume from (default: empty)')
    parser.add_argument('--level', type=int, default=0,
                        help='Optional: Level number to resume from (default: 0)')
    # Add a force flag to force run all games regardless of existing files
    parser.add_argument('--force', action='store_true',
                        help='Force run all games regardless of existing result files')
    parser.add_argument('--run_id_start', type=int, default=1,)
    parser.add_argument('--think_aloud', action='store_true',
                        help='Allow the LLM to think aloud as opposed to outputting strictly the next action.')
    parser.add_argument('--memory', type=int, default=0, help="Number of previous steps to include in the prompt (default: 0)")
    parser.add_argument('--save_dir', type=str, default="llm_agent_results",
                        help='Root directory to save results (default: llm_agent_results)')
    parser.add_argument('--worker_id', type=str, default='',
                        help='Optional identifier for this worker process')
    parser.add_argument('--state_path', type=str, default='',
                        help='Optional path to shared JSON state for coordinating workers')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel worker processes to spawn (default: 1)')
    args = parser.parse_args()

    # Get the list of games from PRIORITY_GAMES
    game_names = PRIORITY_GAMES.copy()
    
    # Find the starting game in the list
    resume_game_name = args.resume_game_name
    actual_resume_game_name = 'atlas shrank' if resume_game_name == 'atlas_shrank' else resume_game_name
    
    if actual_resume_game_name:
        try:
            start_idx = game_names.index(actual_resume_game_name)
            # Keep only games starting from the resume game
            game_names = game_names[start_idx:]
        except ValueError:
            print(f"Warning: Game '{actual_resume_game_name}' not found in PRIORITY_GAMES list. Starting from the beginning.")
    
    # Process in reverse order if flag is set
    if args.reverse:
        game_names.reverse()
        print(f"Processing games in reverse order, starting with: {game_names[0]}")
    else:
        print(f"Processing games in order, starting with: {game_names[0]}")

    print(f"\n=== Running LLM agent with model: {args.model}, for {len(game_names)} games, ensuring up to {args.num_runs} total runs per level ===")

    # Determine model folder name possibly including memory
    model_folder = args.model if args.memory <= 0 else f"{args.model}_mem-{args.memory}"
    save_dir_main = os.path.join(args.save_dir, model_folder)
    os.makedirs(save_dir_main, exist_ok=True)

    # Starting level from CLI argument
    start_level = args.level
    print(f"Starting from level {start_level}")

    # Collect information for all games
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
    
    # Build all jobs list for parallel coordination
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
        # For single worker, run directly
        worker_loop(args, all_jobs, game_info_map, save_dir_main)
    
    print("\n=== All runs for all games and levels have been processed ===")


if __name__ == "__main__":
    main()
