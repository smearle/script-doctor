import argparse
import os
import re
import json
import jax
from lark import Lark
from env_wrappers import RepresentationWrapper
from env import PSParams
from preprocess_games import PS_LARK_GRAMMAR_PATH, get_tree_from_txt
from LLM_agent import LLMGameAgent
from globals import PRIORITY_GAMES

CUSTOM_GAMES_DIR = "data/scraped_games"

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

def check_run_file_exists(save_dir, model, game_name, run_id, level_index,
                          think_aloud):
    """
    Check if the specified run file exists, supporting two naming formats:
    1. With level marker: model_game_run_X_level_Y.json
    2. Without level marker: model_game_run_X.json (assumed to be level 0)
    
    Args:
        save_dir: Directory to check
        model: Model name
        game_name: Game name
        run_id: Run ID
        level_index: Level index
        
    Returns:
        Boolean indicating if file exists
    """
    # Format the game name by replacing spaces with underscores for file paths
    formatted_game_name = game_name.replace(" ", "_")
    cot_prefix = "CoT_" if think_aloud else ""
    
    # Check filename with level marker
    filename_with_level = f"{model}_{cot_prefix}{formatted_game_name}_run_{run_id}_level_{level_index}.json"
    path_with_level = os.path.join(save_dir, filename_with_level)
    
    # Also check with original game name format (for backward compatibility)
    orig_filename_with_level = f"{model}_{cot_prefix}{game_name}_run_{run_id}_level_{level_index}.json"
    orig_path_with_level = os.path.join(save_dir, orig_filename_with_level)
    
    # For level 0, also check filename without level marker
    if level_index == 0:
        filename_without_level = f"{model}_{cot_prefix}{formatted_game_name}_run_{run_id}.json"
        path_without_level = os.path.join(save_dir, filename_without_level)
        
        orig_filename_without_level = f"{model}_{cot_prefix}{game_name}_run_{run_id}.json"
        orig_path_without_level = os.path.join(save_dir, orig_filename_without_level)
        
        exists = os.path.exists(path_with_level) or os.path.exists(path_without_level) or \
                os.path.exists(orig_path_with_level) or os.path.exists(orig_path_without_level)
        return exists
    
    exists = os.path.exists(path_with_level) or os.path.exists(orig_path_with_level)
    return exists

def get_run_file_path(save_dir, model, game_name, run_id, level_index,
                      think_aloud):
    """
    Get file path for saving run results, always using the format with level marker
    
    Args:
        save_dir: Directory to save to
        model: Model name
        game_name: Game name
        run_id: Run ID
        level_index: Level index
        
    Returns:
        File path string
    """
    # Format the game name by replacing spaces with underscores for file paths
    formatted_game_name = game_name.replace(" ", "_")
    
    filename = (f"{model}_" + \
        ('CoT_' if think_aloud else '') + \
        f"{formatted_game_name}_run_{run_id}_level_{level_index}.json")
    
    return os.path.join(save_dir, filename)

def find_next_available_run_id(save_dir, model, game_name, level_index, initial_run_id):
    """
    Find the next available run ID that doesn't conflict with existing files.
    
    Args:
        save_dir: Directory where run results are saved
        model: Model name
        game_name: Game name
        level_index: Level index
        initial_run_id: The initial run ID we'd like to use
        
    Returns:
        int: The next available run ID
    """
    run_id = initial_run_id
    while check_run_file_exists(save_dir, model, game_name, run_id, level_index):
        print(f"Run {run_id} already exists for Game: {game_name}, Level: {level_index}. Trying next run ID.")
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

        env = RepresentationWrapper(tree, debug=False, print_score=False)

        if not hasattr(env, 'levels') or not env.levels:
            print(f"Error: No levels found for game '{game_name}'. Skipping game.")
            return None
            
        if len(env.levels) == 0:
            print(f"Error: No levels available (env.levels is empty) for game '{game_name}'. Skipping game.")
            return None
            
        # Collect game information
        game_info = {
            "game_name": game_name,
            "game_path": game_path,
            "rules": rules,
            "mapping": mapping,
            "ascii_map": ascii_map,
            "env": env,
            "tree": tree,
            "num_levels": len(env.levels),
            "levels_to_process": list(range(start_level, len(env.levels)))
        }
        
        return game_info
        
    except Exception as e:
        print(f"Error collecting game info for {game_name}: {type(e).__name__}, {e}")
        return None

def process_game_level(agent, game_info, level_index, run_id, save_dir, model,
                       max_steps, think_aloud, force=False):
    """
    Process a specific game level for a specific run ID.
    
    Args:
        agent: LLM agent
        game_info: Dictionary containing game information
        level_index: Level index to process
        run_id: Run ID
        save_dir: Directory to save results
        model: Model name
        max_steps: Maximum steps per episode
        force: Whether to force rerun existing results
        
    Returns:
        Boolean indicating success
    """
    game_name = game_info["game_name"]
    env = game_info["env"]
    rules = game_info["rules"]
    
    print(f"\n=== Processing Game: {game_name}, Level: {level_index}, Run: {run_id} ===")
    
    # Check if this run already exists
    if check_run_file_exists(save_dir, model, game_name, run_id, level_index, think_aloud) and not force:
        print(f"Run {run_id} for Game: {game_name}, Level: {level_index} already exists. Skipping.")
        return True
    
    # Get the path for saving results
    current_run_filepath = get_run_file_path(save_dir, model, game_name, run_id,
                                             level_index, think_aloud)

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
                log_file=log_file,
            )
            
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
        
        # Find the next available run ID (starting from the current run_id)
        available_run_id = find_next_available_run_id(save_dir, model, game_name, level_index, run_id)
        
        if available_run_id != run_id:
            print(f"Original run ID {run_id} already exists. Using run ID {available_run_id} instead.")
            # Update the run ID in the result dictionary
            result["run"] = available_run_id
        
        # Get the file path with the updated run ID
        current_run_filepath = get_run_file_path(save_dir, model, game_name, available_run_id, level_index, think_aloud)
        
        # Save results
        with open(current_run_filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=lambda o: f"<{type(o).__name__} instance>")
        print(f"Result saved to {current_run_filepath} (Run ID: {available_run_id})")
        
        return True
        
    except Exception as e:
        print(f"!!! ERROR during Game: {game_name}, Level: {level_index}, Run: {run_id} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='LLM agent loop experiment (env+rules/ascii/mapping)')
    parser.add_argument('--model', type=str, required=True, choices=['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen', 'deepseek-r1'],
                        help='LLM model alias (4o-mini=4o-mini, o3=O3-mini, gemini=Gemini-2.0, gemini-2.5-pro=Gemini-2.5-Pro, deepseek=DeepSeek, qwen=Qwen)')
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
    parser.add_argument('--save_dir', type=str, default="llm_agent_results",
                        help='Directory to save results (default: llm_agent_results)')
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
    
    agent = LLMGameAgent(model_name=args.model)

    save_dir_main = os.path.join(args.save_dir, args.model)
    os.makedirs(save_dir_main, exist_ok=True)

    # Starting level from CLI argument
    start_level = args.level
    print(f"Starting from level {start_level}")

    # 第一步：收集所有游戏的信息
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
    
    # 第二步：按运行ID遍历所有游戏和关卡
    for run_id in range(args.run_id_start, args.num_runs + args.run_id_start):
        print(f"\n\n========== PROCESSING RUN {run_id} OF {args.num_runs} FOR ALL GAMES ==========\n")
        
        # 处理所有游戏
        for game_idx, game_info in enumerate(all_games_info):
            game_name = game_info["game_name"]
            print(f"\n==== Processing game {game_idx+1}/{len(all_games_info)}: {game_name} (Run {run_id}) ====")
            
            # 处理该游戏的所有关卡
            for level_index in game_info["levels_to_process"]:
                process_game_level(
                    agent=agent,
                    game_info=game_info,
                    level_index=level_index,
                    run_id=run_id,
                    save_dir=save_dir_main,
                    think_aloud=args.think_aloud,
                    model=args.model,
                    max_steps=args.max_steps,
                    force=args.force
                )
    
    print("\n=== All runs for all games and levels have been processed ===")

if __name__ == "__main__":
    main()
