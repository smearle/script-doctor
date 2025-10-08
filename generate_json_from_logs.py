#!/usr/bin/env python3
"""
Generate JSON result files from existing log directories.
This script reconstructs the JSON results from log files when the original JSON generation failed.
Now includes environment replay for accurate rewards and heuristics.

usage:
    python generate_json_from_logs.py --search_dir llm_agent_results
"""

import os
import json
import re
import argparse
from pathlib import Path
import sys

# # Add parent directory to path for imports
# script_dir = Path(__file__).parent
# parent_dir = script_dir.parent
# sys.path.insert(0, str(parent_dir))

import jax
from lark import Lark
from puzzlejax.env_wrappers import RepresentationWrapper
from puzzlejax.env import PSParams
from puzzlejax.preprocess_games import PS_LARK_GRAMMAR_PATH, get_tree_from_txt

CUSTOM_GAMES_DIR = "data/scraped_games"

def extract_action_from_log(log_content):
    """Extract the action ID from log content."""
    # Look for ACTION: followed by a number (CoT format)
    match = re.search(r'ACTION:\s*(\d+)', log_content)
    if match:
        return int(match.group(1))

    # Look for "LLM Response:" followed by just a number (non-CoT format)
    match = re.search(r'LLM Response:\s*(\d+)\s*$', log_content, re.MULTILINE)
    if match:
        return int(match.group(1))

    return None

def extract_game_state_from_log(log_content):
    """Extract the ASCII game state from log content."""
    lines = log_content.split('\n')
    map_section = []
    legend_section = []

    in_map = False
    in_legend = False

    for line in lines:
        if line.startswith('MAP:'):
            in_map = True
            in_legend = False
            continue
        elif line.startswith('LEGEND:'):
            in_legend = True
            in_map = False
            continue
        elif line.strip() == '' and (in_map or in_legend):
            if in_map and map_section:
                break
            if in_legend and legend_section:
                in_legend = False
        elif in_map:
            map_section.append(line)
        elif in_legend:
            legend_section.append(line)

    return legend_section, map_section

def parse_log_directory(log_dir_path):
    """Parse all step logs in a directory and extract information."""
    log_dir = Path(log_dir_path)

    if not log_dir.exists():
        return None

    # Get all step files and sort them
    step_files = sorted([f for f in log_dir.glob('step_*.txt')],
                       key=lambda x: int(x.stem.split('_')[1]))

    if not step_files:
        return None

    actions = []
    initial_ascii = None
    final_ascii = None

    for i, step_file in enumerate(step_files):
        try:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract action
            action = extract_action_from_log(content)
            if action is not None:
                actions.append(action)

            # Extract game state (we'll use the first one as initial, last as final)
            legend, map_section = extract_game_state_from_log(content)

            if i == 0 and legend and map_section:
                initial_ascii = ['LEGEND:'] + legend + ['', 'MAP:'] + map_section

            if legend and map_section:
                final_ascii = ['LEGEND:'] + legend + ['', 'MAP:'] + map_section

        except Exception as e:
            print(f"Error reading {step_file}: {e}")
            continue

    return {
        'actions': actions,
        'initial_ascii': initial_ascii,
        'final_ascii': final_ascii,
        'num_steps': len(step_files)
    }

def parse_filename_info(log_dir_name, parent_path=None):
    """Parse model, game, run, and level info from log directory name and parent path."""
    # Remove '_logs' suffix
    if log_dir_name.endswith('_logs'):
        base_name = log_dir_name[:-5]
    else:
        base_name = log_dir_name

    # Extract model from parent directory if available
    model = None
    if parent_path:
        parent_name = Path(parent_path).name
        if parent_name in ['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen', 'deepseek-r1']:
            model = parent_name
        elif '_mem-' in parent_name:
            # Handle memory variants like "4o-mini_mem-5"
            model_part = parent_name.split('_mem-')[0]
            if model_part in ['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen', 'deepseek-r1']:
                model = model_part

    # Parse the base name for game info
    # New format: [CoT_]game_run_X_level_Y
    # Old format: model_[CoT_]game_run_X_level_Y
    parts = base_name.split('_')

    game_parts = []
    run_id = None
    level_id = None
    cot_mode = False

    i = 0
    while i < len(parts):
        if parts[i] in ['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen', 'deepseek-r1']:
            # Old format - model in filename
            if not model:
                model = parts[i]
        elif parts[i] == 'CoT':
            cot_mode = True
        elif parts[i] == 'run' and i + 1 < len(parts):
            run_id = int(parts[i + 1])
            i += 1  # Skip the number
        elif parts[i] == 'level' and i + 1 < len(parts):
            level_id = int(parts[i + 1])
            i += 1  # Skip the number
        else:
            # This must be part of the game name
            if parts[i] not in ['run', 'level', 'CoT'] and parts[i] not in ['4o-mini', 'o3-mini', 'gemini', 'gemini-2.5-pro', 'deepseek', 'qwen', 'deepseek-r1']:
                game_parts.append(parts[i])
        i += 1

    # Join game parts with underscores, but the original game name might use spaces
    game_name = '_'.join(game_parts) if game_parts else 'unknown'

    return {
        'model': model,
        'game': game_name,
        'run': run_id,
        'level': level_id,
        'think_aloud': cot_mode
    }

def create_game_environment(game_name):
    """Create and return game environment for the specified game."""
    try:
        # Find game file - try both original and space-replaced versions
        game_path = None
        possible_names = [game_name, game_name.replace('_', ' ')]

        for name in possible_names:
            potential_path = os.path.join(CUSTOM_GAMES_DIR, f"{name}.txt")
            if os.path.exists(potential_path):
                game_path = potential_path
                break

        if not game_path:
            print(f"Game file not found for: {game_name}")
            return None

        # Initialize environment parser
        with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
            puzzlescript_grammar = f.read()
        grammar_parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)

        # Use the actual game name from the file for parsing
        actual_game_name = os.path.basename(game_path)[:-4]  # Remove .txt extension
        tree, success, err_msg = get_tree_from_txt(grammar_parser, actual_game_name, test_env_init=False)

        if success != 0:
            print(f"Failed to parse game file for {game_name}: {err_msg}")
            return None

        env = RepresentationWrapper(tree, debug=False, print_score=False)

        if not hasattr(env, 'levels') or not env.levels:
            print(f"No levels found for game '{game_name}'")
            return None

        return env

    except Exception as e:
        print(f"Error creating environment for {game_name}: {type(e).__name__}, {e}")
        return None

def replay_actions_in_environment(env, level_index, actions, run_id):
    """Replay actions in the environment to get accurate rewards and heuristics."""
    try:
        level = env.get_level(level_index)
        env_params = PSParams(level=level)
        # Use run_id as seed to ensure same initial state as original run
        rng = jax.random.PRNGKey(run_id * 1000 + level_index)

        obs, state = env.reset(rng=rng, params=env_params)
        initial_ascii = env.render_ascii_and_legend(state).split('\n')

        rewards = []
        heuristics = []
        current_state = state

        for action_id in actions:
            rng, _rng = jax.random.split(rng)
            obs, next_state, reward, done, info = env.step_env(
                rng=rng, action=action_id, state=current_state, params=env_params
            )

            rewards.append(float(reward.item()))
            heuristics.append(float(next_state.heuristic.item()))
            current_state = next_state

            if done:
                break

        final_ascii = env.render_ascii_and_legend(current_state).split('\n')

        return {
            'initial_ascii': initial_ascii,
            'final_ascii': final_ascii,
            'rewards': rewards,
            'heuristics': heuristics,
            'final_state': {
                'score': int(current_state.score),
                'win': bool(current_state.win),
                'step': len(actions)
            }
        }

    except Exception as e:
        print(f"Error replaying actions: {type(e).__name__}, {e}")
        return None

def generate_json_from_logs(log_dir_path, output_dir=None):
    """Generate a JSON result file from log directory."""
    log_dir = Path(log_dir_path)

    # Parse filename info, including parent directory for model info
    # filename_info = parse_filename_info(log_dir.name)
    filename_info = parse_filename_info(log_dir.name, log_dir.parent)

    # Parse log content
    log_data = parse_log_directory(log_dir)

    if not log_data:
        print(f"Could not parse log data from {log_dir}")
        return None

    # Try to replay actions in environment for accurate data
    env = create_game_environment(filename_info['game'])
    replay_data = None

    if env and log_data['actions']:
        print(f"Replaying {len(log_data['actions'])} actions in environment...")
        replay_data = replay_actions_in_environment(
            env, filename_info['level'], log_data['actions'], filename_info['run']
        )

    # Create result structure
    if replay_data:
        # Use accurate data from environment replay
        result = {
            "model": filename_info['model'],
            "game": filename_info['game'],
            "level": filename_info['level'],
            "run": filename_info['run'],
            "win": replay_data['final_state']['win'],
            "action_sequence": log_data['actions'],
            "reward_sequence": replay_data['rewards'],
            "initial_ascii": replay_data['initial_ascii'],
            "final_ascii": replay_data['final_ascii'],
            "state_data": replay_data['final_state'],
            "heuristics": replay_data['heuristics'],
            "_generated_from_logs": True,
            "_replayed_in_environment": True,
            "_note": "This JSON was reconstructed from logs with environment replay for accurate rewards/heuristics."
        }
    else:
        # Fallback to default values if environment replay failed
        print("Environment replay failed, using default values...")
        result = {
            "model": filename_info['model'],
            "game": filename_info['game'],
            "level": filename_info['level'],
            "run": filename_info['run'],
            "win": False,  # Cannot determine from logs alone
            "action_sequence": log_data['actions'],
            "reward_sequence": [-0.01] * len(log_data['actions']),  # Default reward
            "initial_ascii": log_data['initial_ascii'] or [],
            "final_ascii": log_data['final_ascii'] or [],
            "state_data": {
                "score": 0,  # Cannot determine from logs
                "win": False,  # Cannot determine from logs
                "step": log_data['num_steps']
            },
            "heuristics": [0.0] * len(log_data['actions']),  # Cannot determine from logs
            "_generated_from_logs": True,
            "_note": "This JSON was reconstructed from logs. Environment replay failed, using default values."
        }

    # Determine output path
    if output_dir is None:
        output_dir = log_dir.parent
    else:
        output_dir = Path(output_dir)

    # Generate filename 
    cot_prefix = "CoT_" if filename_info['think_aloud'] else ""
    json_filename = f"{filename_info['model']}_{cot_prefix}{filename_info['game']}_run_{filename_info['run']}_level_{filename_info['level']}.json"
    output_path = output_dir / json_filename

    # Write JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"Generated JSON: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error writing JSON file {output_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate JSON result files from log directories')
    parser.add_argument('--log_dir', type=str, help='Specific log directory to process')
    parser.add_argument('--search_dir', type=str, default='llm_agent_results',
                       help='Directory to search for log directories (default: llm_agent_results)')
    parser.add_argument('--output_dir', type=str, help='Output directory for JSON files (default: same as log directory)')

    args = parser.parse_args()

    if args.log_dir:
        # Process specific log directory
        generate_json_from_logs(args.log_dir, args.output_dir)
    else:
        # Search for all log directories
        search_path = Path(args.search_dir)
        if not search_path.exists():
            print(f"Search directory {search_path} does not exist")
            return

        log_dirs = []
        for root, dirs, files in os.walk(search_path):
            for dirname in dirs:
                if dirname.endswith('_logs'):
                    log_path = Path(root) / dirname
                    # Check if corresponding JSON already exists
                    filename_info = parse_filename_info(dirname)
                    if filename_info['model'] and filename_info['run'] is not None and filename_info['level'] is not None:
                        cot_prefix = "CoT_" if filename_info['think_aloud'] else ""
                        json_filename = f"{filename_info['model']}_{cot_prefix}{filename_info['game']}_run_{filename_info['run']}_level_{filename_info['level']}.json"
                        json_path = Path(root) / json_filename

                        if not json_path.exists():
                            log_dirs.append(log_path)

        print(f"Found {len(log_dirs)} log directories without corresponding JSON files")

        success_count = 0
        for log_dir in log_dirs:
            if generate_json_from_logs(log_dir, args.output_dir):
                success_count += 1

        print(f"Successfully generated {success_count}/{len(log_dirs)} JSON files")

if __name__ == "__main__":
    main()