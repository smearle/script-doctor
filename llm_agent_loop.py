import argparse
import os
import re
import json
import jax
from lark import Lark
from wrappers import RepresentationWrapper
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

def check_run_file_exists(save_dir, model, game_name, run_id, level_index):
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
    
    # Check filename with level marker
    filename_with_level = f"{model}_{formatted_game_name}_run_{run_id}_level_{level_index}.json"
    path_with_level = os.path.join(save_dir, filename_with_level)
    
    # Also check with original game name format (for backward compatibility)
    orig_filename_with_level = f"{model}_{game_name}_run_{run_id}_level_{level_index}.json"
    orig_path_with_level = os.path.join(save_dir, orig_filename_with_level)
    
    # For level 0, also check filename without level marker
    if level_index == 0:
        filename_without_level = f"{model}_{formatted_game_name}_run_{run_id}.json"
        path_without_level = os.path.join(save_dir, filename_without_level)
        
        orig_filename_without_level = f"{model}_{game_name}_run_{run_id}.json"
        orig_path_without_level = os.path.join(save_dir, orig_filename_without_level)
        
        exists = os.path.exists(path_with_level) or os.path.exists(path_without_level) or \
                os.path.exists(orig_path_with_level) or os.path.exists(orig_path_without_level)
        return exists
    
    exists = os.path.exists(path_with_level) or os.path.exists(orig_path_with_level)
    return exists

def get_run_file_path(save_dir, model, game_name, run_id, level_index):
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
    
    filename = f"{model}_{formatted_game_name}_run_{run_id}_level_{level_index}.json"
    return os.path.join(save_dir, filename)

def get_existing_and_missing_runs(save_dir, model, game_name, level_index, num_runs):
    """
    Determine which runs exist and which are missing for a specific game level
    
    Args:
        save_dir: Directory to check
        model: Model name
        game_name: Game name
        level_index: Level index
        num_runs: Total number of runs to check
        
    Returns:
        Tuple of (existing_runs, missing_runs) lists
    """
    existing_runs = []
    missing_runs = []
    
    for run_id in range(1, num_runs + 1):
        if check_run_file_exists(save_dir, model, game_name, run_id, level_index):
            existing_runs.append(run_id)
        else:
            missing_runs.append(run_id)
            
    return existing_runs, missing_runs

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

def main():
    parser = argparse.ArgumentParser(description='LLM agent loop experiment (env+rules/ascii/mapping)')
    parser.add_argument('--model', type=str, required=True, choices=['4o-mini', 'o3-mini', 'gemini', 'deepseek', 'qwen'],
                        help='LLM model alias (4o-mini=4o-mini, o3=O3-mini, gemini=Gemini-2.0, deepseek=DeepSeek, qwen=Qwen)')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode (default: 100)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs per game (default: 10)')
    parser.add_argument('--reverse', action='store_true',
                        help='Process games in reverse order')
    parser.add_argument('--resume_game_name', type=str, default='',
                        help='Name of the game to resume from (default: atlas_shrank)')
    parser.add_argument('--level', type=int, default=0,
                        help='Optional: Level number to resume from (default: 0)')
    # Add a force flag to force run all games regardless of existing files
    parser.add_argument('--force', action='store_true',
                        help='Force run all games regardless of existing result files')
    args = parser.parse_args()

    # Get the list of games from PRIORITY_GAMES
    game_names = PRIORITY_GAMES.copy()
    
    # Find the starting game in the list
    resume_game_name = args.resume_game_name
    actual_resume_game_name = 'atlas shrank' if resume_game_name == 'atlas_shrank' else resume_game_name
    
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
    
    action_space = [0, 1, 2, 3, 4]
    action_meanings = {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}

    print(f"\n=== Running LLM agent with model: {args.model}, for {len(game_names)} games, ensuring up to {args.num_runs} total runs per level ===")
    if args.reverse:
        print("Processing games in reverse order")
    
    agent = LLMGameAgent(model_name=args.model)

    save_dir_main = "llm_agent_results"
    os.makedirs(save_dir_main, exist_ok=True)

    # Starting level from CLI argument
    start_level = args.level
    print(f"Starting from level {start_level}")

    # Process each game in the list
    for game_idx, game_name in enumerate(game_names):
        print(f"\n==== Processing game {game_idx+1}/{len(game_names)}: {game_name} ====")
        
        game_path = os.path.join(CUSTOM_GAMES_DIR, f"{game_name}.txt")
        if not os.path.exists(game_path):
            print(f"Error: Game file not found at {game_path}. Skipping game.")
            continue

        # Extract sections for LLM agent
        rules_lines = extract_section(game_path, "RULES")
        legend_lines = extract_section(game_path, "LEGEND")
        level_lines = extract_section(game_path, "LEVELS")
        rules = "\n".join(rules_lines)
        mapping = parse_legend(legend_lines)
        ascii_map = extract_first_level(level_lines)

        if not rules or not mapping or not ascii_map:
            print(f"Error: Missing rules, mapping, or initial level ASCII for game {game_name}. Skipping game.")
            continue

        # Print game info
        print(f"\n--- Preparing game: {game_name} ---")

        # Initialize environment parser
        with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
            puzzlescript_grammar = f.read()
        grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
        tree, success, err_msg = get_tree_from_txt(grammar_parser, game_name, test_env_init=False)
        if success != 0:
            print(f"Error: Failed to parse game file for {game_name}. Skipping game.")
            print(f"Parse error: {err_msg}, code: {success}")
            continue

        env = RepresentationWrapper(tree, debug=False, print_score=False)

        if not hasattr(env, 'levels') or not env.levels:
            print(f"Error: No levels found for game '{game_name}'. Skipping game.")
            continue
            
        if len(env.levels) == 0:
            print(f"Error: No levels available (env.levels is empty) for game '{game_name}'. Skipping game.")
            continue

        # Process each level starting from the specified resume level
        for level_index in range(start_level, len(env.levels)):
            print(f"\n=== Processing Game: {game_name}, Level: {level_index} ===")
            
            # Continue processing this level until all runs are complete (or processed once in force mode)
            while True:
                # Re-check which runs exist and which are missing at the beginning of each loop
                existing_runs, missing_runs = get_existing_and_missing_runs(
                    save_dir_main, args.model, game_name, level_index, args.num_runs
                )
                
                # Print detailed information
                if existing_runs:
                    print(f"Found existing runs: {existing_runs}")
                
                # Determine if we need to continue processing this level
                if not missing_runs and not args.force:
                    print(f"All {args.num_runs} runs complete for this level. Moving to next level.")
                    break  # Exit the while loop, move to next level
                elif args.force and len(existing_runs) == args.num_runs:
                    # If in force mode and all runs have been processed once, complete
                    print(f"Force mode: All {args.num_runs} runs have been processed once. Moving to next level.")
                    break  # Exit the while loop, move to next level
                
                # Determine the next run ID to process
                if args.force:
                    # In force mode, prioritize missing runs; otherwise select first existing run to reprocess
                    target_run_id = missing_runs[0] if missing_runs else existing_runs[0]
                else:
                    # In non-force mode, only process missing runs
                    target_run_id = missing_runs[0]
                
                print(f"\n--- Processing Game: {game_name}, Level: {level_index} (Run ID: {target_run_id}/{args.num_runs}) ---")
                
                # Get the path for saving results
                current_run_filepath = get_run_file_path(save_dir_main, args.model, game_name, target_run_id, level_index)
                
                # Try to process one run
                try:
                    # Ensure level_index is valid
                    if level_index < 0 or level_index >= len(env.levels):
                        print(f"Warning: Invalid level_index {level_index} for game '{game_name}' (max index {len(env.levels)-1}). Skipping level.")
                        break  # Exit from the while loop
                    
                    level = env.get_level(level_index)
                    env_params = PSParams(level=level)
                    # Use target_run_id as seed to ensure uniqueness
                    rng = jax.random.PRNGKey(target_run_id * 1000 + level_index)
                    
                    obs, state = env.reset(rng=rng, params=env_params)
                    ascii_state = env.render_ascii_and_legend(state)
                    
                    result = {
                        "model": args.model,
                        "game": game_name,
                        "level": level_index,
                        "run": target_run_id,
                        "win": False,
                        "action_sequence": [],
                        "reward_sequence": [],
                        "heuristic_sequence": []
                    }
                    
                    state_history = set()
                    current_state = state
                    
                    # Main action loop
                    for step in range(args.max_steps):
                        print(f"\nStep {step+1}/{args.max_steps}")
                        
                        h = hash(current_state.multihot_level.tobytes())
                        state_history.add(h)
                        
                        action_id = agent.choose_action(
                            ascii_map=ascii_state,
                            rules=rules,
                            action_space=action_space,
                            action_meanings=action_meanings,
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
                    
                    # Find the next available run ID (starting from the current target_run_id)
                    available_run_id = find_next_available_run_id(save_dir_main, args.model, game_name, level_index, target_run_id)
                    
                    if available_run_id != target_run_id:
                        print(f"Original run ID {target_run_id} already exists. Using run ID {available_run_id} instead.")
                        # Update the run ID in the result dictionary
                        result["run"] = available_run_id
                    
                    # Get the file path with the updated run ID
                    current_run_filepath = get_run_file_path(save_dir_main, args.model, game_name, available_run_id, level_index)
                    
                    # Save results
                    with open(current_run_filepath, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, default=lambda o: f"<{type(o).__name__} instance>")
                    print(f"Result saved to {current_run_filepath} (Run ID: {available_run_id})")
                    
                except Exception as e:
                    print(f"!!! ERROR during Game: {game_name}, Level: {level_index} (Run ID: {target_run_id}) !!!")
                    print(f"Error type: {type(e).__name__}, Message: {e}")
                    # If an error occurs during processing, continue with other runs
                    continue

    print("\n=== All runs for all games and levels have been processed ===")

if __name__ == "__main__":
    main()