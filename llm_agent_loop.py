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

def main():
    parser = argparse.ArgumentParser(description='LLM agent loop experiment (env+rules/ascii/mapping)')
    parser.add_argument('--model', type=str, required=True, choices=['4o-mini', 'o3-mini', 'gemini', 'deepseek', 'qwen'],
                        help='LLM model alias (4o-mini=4o-mini, o3=O3-mini, gemini=Gemini-2.0, deepseek=DeepSeek, qwen=Qwen)')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode (default: 100)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs per game (default: 10)')
    args = parser.parse_args()

    game_names = PRIORITY_GAMES
    action_space = [0, 1, 2, 3, 4]
    action_meanings = {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}

    print(f"\n=== Running LLM agent with model: {args.model} for {args.num_runs} runs per game ===")
    agent = LLMGameAgent(model_name= args.model)

    for run_idx in range(args.num_runs): # Outer loop for multiple runs
        print(f"\n--- STARTING RUN {run_idx + 1}/{args.num_runs} ---")
        for game_name in game_names:
            game_path = os.path.join(CUSTOM_GAMES_DIR, f"{game_name}.txt")
            if not os.path.exists(game_path):
                print(f"Skipping {game_name}: File not found.")
                continue

            # Extract sections for LLM agent
            rules_lines = extract_section(game_path, "RULES")
            legend_lines = extract_section(game_path, "LEGEND")
            level_lines = extract_section(game_path, "LEVELS")
            rules = "\n".join(rules_lines)
            mapping = parse_legend(legend_lines)
            ascii_map = extract_first_level(level_lines)

            if not rules or not mapping or not ascii_map:
                print(f"Skipping {game_name}: Missing rules, mapping, or level.")
                continue

            print(f"\n--- Playing game: {game_name} (Run {run_idx + 1}) ---")
            print("Extracted RULES section:")
            print(rules)
            print("Legend mapping (per game):")
            for k, v in mapping.items():
                print(f"{repr(k)}: {', '.join(v)}")
            print("Initial game state (ASCII):")
            print(ascii_map)

            # Initialize environment
            with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
                puzzlescript_grammar = f.read()
            # parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
            grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
            tree, success, err_msg = get_tree_from_txt(grammar_parser, game_name, test_env_init=False)
            if success != 0:
                print(f"Skipping {game_name}: Failed to parse game file.")
                print(f"Parse error: {err_msg}, code: {success}")
                continue

            env = RepresentationWrapper(tree, debug=False, print_score=False)
            for level_index in range(1, len(env.levels)):
                level = env.get_level(level_index)
                env_params = PSParams(level=level)
                rng = jax.random.PRNGKey(run_idx + level_index)

                obs, state = env.reset(rng=rng, params=env_params)
                ascii_state = env.render_ascii_and_legend(state)

                result = {
                    "model": args.model,
                    "game": game_name,
                    "win": False,
                    "action_sequence": [],
                    "reward_sequence": [],
                    "score_sequence": [],
                    "heuristic_sequence": []
                }

                state_history = set()
                current_state = state
                current_mapping = env.get_ascii_mapping()
                current_legend = current_mapping
                for step in range(args.max_steps):
                    print(f"\nStep {step+1}/{args.max_steps}")

                    h = hash(state.multihot_level.tobytes())
                    state_history.add(h)

                    # Get dynamically updated ASCII mapping info
    

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

                    # Extract actual reward values from JAX array and convert types
                    result["reward_sequence"].append(float(rew.item()))  # 使用.item()提取标量值
                    
                    # Record initial and final ASCII states
                    if step == 0:
                        result["initial_ascii"] = ascii_state.split('\n')  # 存储为列表方便阅读
                    result["final_ascii"] = ascii_state.split('\n')  # 每次更新最终状态

                    # Record raw heuristic values to sequence
                    result["heuristic_sequence"].append(float(next_state.heuristic.item()))  # 确保提取标量值
                    
                    # Simplify state data
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

                # Preserve and convert heuristic data
                result["heuristics"] = [float(h) for h in result["heuristic_sequence"]]
                del result["heuristic_sequence"]
                del result["score_sequence"] 
                
                # Ensure final ASCII state is from last step
                result["final_ascii"] = ascii_state.split('\n')
                
                save_dir = "llm_agent_results"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{args.model}_{game_name}_run_{run_idx + 1}_level_{level_index}.json") # Added run_idx
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, default=lambda o: f"<{type(o).__name__} instance>")
                print(f"Result saved to {save_path}")

if __name__ == "__main__":
    main()
