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
    mapping = {}
    for line in legend_lines:
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            objs = [obj.strip() for obj in v.strip().split()]
            mapping[k] = objs
    return mapping

def extract_first_level(level_lines):
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
    检查指定的运行文件是否存在，支持两种命名格式：
    1. 带level标记: model_game_run_X_level_Y.json
    2. 不带level标记: model_game_run_X.json (假定为level 0)
    """
    # 检查带level标记的文件名
    filename_with_level = f"{model}_{game_name}_run_{run_id}_level_{level_index}.json"
    path_with_level = os.path.join(save_dir, filename_with_level)
    
    # 对于level 0，还检查不带level标记的文件名
    if level_index == 0:
        filename_without_level = f"{model}_{game_name}_run_{run_id}.json"
        path_without_level = os.path.join(save_dir, filename_without_level)
        return os.path.exists(path_with_level) or os.path.exists(path_without_level)
    
    return os.path.exists(path_with_level)

def get_run_file_path(save_dir, model, game_name, run_id, level_index):
    """
    获取保存运行结果的文件路径，始终使用带level标记的格式
    """
    filename = f"{model}_{game_name}_run_{run_id}_level_{level_index}.json"
    return os.path.join(save_dir, filename)

def main():
    parser = argparse.ArgumentParser(description='LLM agent loop experiment (env+rules/ascii/mapping)')
    parser.add_argument('--model', type=str, required=True, choices=['4o-mini', 'o3-mini', 'gemini', 'deepseek', 'qwen'],
                        help='LLM model alias (4o-mini=4o-mini, o3=O3-mini, gemini=Gemini-2.0, deepseek=DeepSeek, qwen=Qwen)')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode (default: 100)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs per game (default: 10)')
    parser.add_argument('--resume_game_name', type=str, default=None,
                        help='Optional: Name of the game to resume from.')
    parser.add_argument('--resume_level_num', type=int, default=0,
                        help='Optional: Level number to resume from for the specified resume_game_name (default: 0). Only used if --resume_game_name is provided.')
    args = parser.parse_args()

    game_names = PRIORITY_GAMES
    action_space = [0, 1, 2, 3, 4]
    action_meanings = {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}

    print(f"\n=== Running LLM agent with model: {args.model}, ensuring up to {args.num_runs} total runs per game/level ===")
    agent = LLMGameAgent(model_name=args.model)

    save_dir_main = "llm_agent_results"
    os.makedirs(save_dir_main, exist_ok=True)

    # --- Configuration for starting point (controlled by CLI args) ---
    TARGET_START_GAME_NAME = None
    TARGET_START_LEVEL_FOR_START_GAME = 0  # 默认从level 0开始
    start_game_idx_in_priority_list = -1  # Default: run all games

    if args.resume_game_name:
        TARGET_START_GAME_NAME = args.resume_game_name
        TARGET_START_LEVEL_FOR_START_GAME = args.resume_level_num
        if PRIORITY_GAMES:
            try:
                start_game_idx_in_priority_list = PRIORITY_GAMES.index(TARGET_START_GAME_NAME)
                print(f"Resuming from game: '{TARGET_START_GAME_NAME}' (index {start_game_idx_in_priority_list}), level: {TARGET_START_LEVEL_FOR_START_GAME}")
            except ValueError:
                print(f"WARNING: resume_game_name '{TARGET_START_GAME_NAME}' not found in PRIORITY_GAMES.")
                print("The script will run all games from their beginning.")
                TARGET_START_GAME_NAME = None  # Reset to ensure default behavior
                start_game_idx_in_priority_list = -1
        else:
            print("WARNING: PRIORITY_GAMES list is empty. Cannot use resume_game_name.")
            TARGET_START_GAME_NAME = None  # Reset
            start_game_idx_in_priority_list = -1
    else:
        print(f"No resume_game_name specified. Running all games from level {TARGET_START_LEVEL_FOR_START_GAME}.")

    # Single pass to process games/levels
    if not game_names:
        print("No games to process.")
        return  # Exit if no games

    for current_game_idx, game_name in enumerate(game_names):
        if start_game_idx_in_priority_list != -1 and current_game_idx < start_game_idx_in_priority_list:
            print(f"Skipping game '{game_name}' (comes before resume target '{TARGET_START_GAME_NAME}').")
            continue

        game_path = os.path.join(CUSTOM_GAMES_DIR, f"{game_name}.txt")
        if not os.path.exists(game_path):
            print(f"Skipping {game_name}: File not found at {game_path}.")
            continue

        # Extract sections for LLM agent
        rules_lines = extract_section(game_path, "RULES")
        legend_lines = extract_section(game_path, "LEGEND")
        level_lines = extract_section(game_path, "LEVELS")
        rules = "\n".join(rules_lines)
        mapping = parse_legend(legend_lines)
        ascii_map = extract_first_level(level_lines)

        if not rules or not mapping or not ascii_map:
            print(f"Skipping {game_name}: Missing rules, mapping, or initial level ASCII.")
            continue

        # Print game-level info once before iterating through its levels/runs
        print(f"\n--- Preparing game: {game_name} ---")

        # Initialize environment parser once per game
        with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
            puzzlescript_grammar = f.read()
        grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
        tree, success, err_msg = get_tree_from_txt(grammar_parser, game_name, test_env_init=False)
        if success != 0:
            print(f"Skipping {game_name}: Failed to parse game file.")
            print(f"Parse error: {err_msg}, code: {success}")
            continue

        env = RepresentationWrapper(tree, debug=False, print_score=False)

        # Determine starting level for the current game
        actual_start_level_for_current_game = 0  # 默认从level 0开始
        # Apply specific start level only if TARGET_START_GAME_NAME is set and the current game is that target game.
        if TARGET_START_GAME_NAME and game_name == TARGET_START_GAME_NAME and start_game_idx_in_priority_list != -1:
            actual_start_level_for_current_game = TARGET_START_LEVEL_FOR_START_GAME
        
        if not hasattr(env, 'levels') or not env.levels:
            print(f"Game '{game_name}': No levels found or 'env.levels' is missing. Skipping game.")
            continue
            
        if len(env.levels) == 0:  # No levels at all
            print(f"Game '{game_name}': No levels available (env.levels is empty). Skipping game.")
            continue

        # 检查level从0开始，而不是从1开始
        for level_index in range(actual_start_level_for_current_game, len(env.levels)):
            # 检查是否所有目标运行都已存在
            all_runs_exist = True
            if args.num_runs > 0:
                for run_id in range(1, args.num_runs + 1):
                    if not check_run_file_exists(save_dir_main, args.model, game_name, run_id, level_index):
                        all_runs_exist = False
                        break

            if all_runs_exist:
                print(f"--- Game: {game_name}, Level: {level_index} --- All {args.num_runs} target runs already exist. Skipping this level.")
                continue  # Move to the next level_index

            # Attempt to generate missing runs from 1 up to args.num_runs for this level
            for target_run_id in range(1, args.num_runs + 1):
                # 检查是否已存在，包括两种可能的命名格式
                if check_run_file_exists(save_dir_main, args.model, game_name, target_run_id, level_index):
                    continue  # This run already exists, check the next target_run_id

                # 获取保存结果的路径（始终使用带level的格式）
                current_run_filepath = get_run_file_path(save_dir_main, args.model, game_name, target_run_id, level_index)
                
                # If we are here, the file for target_run_id does not exist, so we need to generate it.
                print(f"\n--- Processing Game: {game_name}, Level: {level_index} (Targeting Run ID: {target_run_id}/{args.num_runs}) ---")
                
                # --- Start of TRY block for robustness for a single run generation ---
                try:
                    # 确保level_index有效
                    if level_index < 0 or level_index >= len(env.levels):  # Modified to allow level 0
                        print(f"Warning: Invalid level_index {level_index} for game '{game_name}' (max index {len(env.levels)-1}). Skipping level for Run ID {target_run_id}.")
                        break  # from target_run_id loop

                    level = env.get_level(level_index)
                    env_params = PSParams(level=level)
                    # Seed using the target_run_id to ensure unique seeds for unique saved runs
                    rng = jax.random.PRNGKey(target_run_id * 1000 + level_index) 

                    obs, state = env.reset(rng=rng, params=env_params)
                    ascii_state = env.render_ascii_and_legend(state)

                    result = {
                        "model": args.model,
                        "game": game_name,
                        "level": level_index,  # Store level index
                        "run": target_run_id,   # Store the target run ID being generated
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

                        h = hash(current_state.multihot_level.tobytes())  # Use current_state
                        state_history.add(h)  # Track state history to detect cycles

                        action_id = agent.choose_action(
                            ascii_map=ascii_state,  # This is the current dynamic state
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
                        ascii_state = env.render_ascii_and_legend(next_state)  # Update ascii_state for next iteration
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
                        if done:  # and not next_state.win
                            print(f"Game ended in {step+1} steps (done=True, not win).")
                            break

                        current_state = next_state

                    # Process final results
                    result["heuristics"] = [float(h) for h in result["heuristic_sequence"]]
                    if "heuristic_sequence" in result: 
                        del result["heuristic_sequence"]
                    result["final_ascii"] = ascii_state.split('\n')  # Ensure it's the very last state
                    
                    # Save results
                    with open(current_run_filepath, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, default=lambda o: f"<{type(o).__name__} instance>")
                    print(f"Result saved to {current_run_filepath}")

                except Exception as e:
                    print(f"!!! ERROR during Game: {game_name}, Level: {level_index} (Targeting Run ID: {target_run_id}) !!!")
                    print(f"Error type: {type(e).__name__}, Message: {e}")
                    # import traceback
                    # print(traceback.format_exc())
                    print("Skipping further run attempts for this level due to error.")
                    break  # Breaks from the target_run_id loop for the current level_index

if __name__ == "__main__":
    main()