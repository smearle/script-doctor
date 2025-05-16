import json
import os

import hydra
from matplotlib import pyplot as plt

from conf.config import PlotRandProfileConfig
from globals import PLOTS_DIR, GAMES_TO_N_RULES_PATH
from profile_rand_jax import JAX_N_ENVS_TO_FPS_PATH, get_step_int, get_level_int
from globals import STANDALONE_NODEJS_RESULTS_PATH


GAMES_TO_PLOT = [
    'sokoban_basic',
    'sokoban_match3',
    'limerick',
    'blocks',
    'slidings',
    'notsnake',
    'Travelling_salesman',
    'Zen_Puzzle_Garden',
    # 'Multi-word_Dictionary_Game',
    'Take_Heart_Lass',
]


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_rand_profile_config")
def main(cfg: PlotRandProfileConfig):
    with open(JAX_N_ENVS_TO_FPS_PATH, 'r') as f:
        results = json.load(f)

    with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
        results_standalone = json.load(f)

    with open(GAMES_TO_N_RULES_PATH, 'r') as f:
        games_to_n_rules = json.load(f)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    devices = results.keys()
    for device in devices:
        print(f'Device: {device}')
        rollout_len_str = results[device].keys()
        for rollout_len_str in rollout_len_str:
            print(f'Rollout len: {rollout_len_str}')

            if cfg.all_games:
                games = results[device][rollout_len_str].keys()
            else:
                games = GAMES_TO_PLOT

            games_n_rules = []
            for game in games:
                if game not in games_to_n_rules:
                    print(f'Game {game} not found in games_to_n_rules. You may need to run sort_games_by_n_rules.py first.')
                    continue
                games_n_rules.append((game, games_to_n_rules[game]))

            games_n_rules = sorted(games_n_rules, key=lambda x: x[1][0])

            n_games = len(games)
            # Make a square grid of subplots
            n_rows = int(n_games ** 0.5)
            n_cols = int(n_games / n_rows) + (n_games % n_rows > 0)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5)) 

            if n_games == 1:
                axes = [axes]

            for game_i, (game, (n_rules, stochastic)) in enumerate(games_n_rules):
                (ax_x, ax_y) = (game_i // n_cols, game_i % n_cols)
                if n_rows == 1:
                    ax = axes[game_i]
                else:
                    ax = axes[ax_x, ax_y]

                levels = results[device][rollout_len_str][game].keys()
                for level_str in levels:
                    level_i = get_level_int(level_str)
                    if level_i != 0:
                        print(f"Ignoring levels other than 0 for now ({game})")
                        continue

                    n_envs_to_fps = results[device][rollout_len_str][game][level_str]

                    n_envs = list(n_envs_to_fps.keys())
                    n_envs = [int(n_env) for n_env in n_envs]
                    fpss = list(n_envs_to_fps.values())
                    fps = [f[-1] for f in fpss]
                    sorted_idxs = sorted(range(len(n_envs)), key=lambda k: int(n_envs[k]))
                    n_envs = [n_envs[i] for i in sorted_idxs]
                    fps = [fps[i] for i in sorted_idxs]
                    
                    ax.plot(n_envs, fps, label=level_str, marker='x', markersize=5, linestyle='-')

                    # Make the y-axis logarithmic
                    ax.set_yscale('linear')
                    ax.set_xscale('linear')
                    ax.set_xlabel('batch size')
                    ax.set_ylabel('FPS')
                    ax.grid(True)
                    n_rules, has_randomness = games_to_n_rules[game]
                    ax.set_title(f'{game}\n({n_rules} rule{"s" if n_rules != 1 else ""}{", stochastic" if has_randomness else ""})')

                    print(f'Game: {game}')

                    # Plot each of the random rollout FPS's from nodejs as broken lines running horizontally
                    
                    if game in results_standalone['randomRollout']:
                        if str(level_i) in results_standalone['randomRollout'][game]:
                            if "Error" in results_standalone['randomRollout'][game][str(level_i)]:
                                print(f'Error in nodejs results for game {game} level {level_i}: {results_standalone["randomRollout"][game][str(level_i)]["Error"]}')
                            else:
                                nodejs_rand_rollout_fps = results_standalone['randomRollout'][game][str(level_i)]['FPS']
                                ax.axhline(y=nodejs_rand_rollout_fps, color='r', linestyle='--', label='NodeJS FPS')
                        else:
                            print(f'Level {level_i} not found in nodejs results for game {game}')
                    else:
                        print(f'Game {game} not found in nodejs results')
                    if game in results_standalone['rand_rollout_from_python']:
                        if str(level_i) in results_standalone['rand_rollout_from_python'][game]:
                            if "Error" in results_standalone['rand_rollout_from_python'][game][str(level_i)]:
                                print(f'Error in nodejs results for game {game} level {level_i}: {results_standalone["rand_rollout_from_python"][game][str(level_i)]["Error"]}')
                            else:
                                nodejs_rand_rollout_python_fps = results_standalone['rand_rollout_from_python'][game][str(level_i)]['FPS']
                                ax.axhline(y=nodejs_rand_rollout_python_fps, color='g', linestyle='--', label='Python-NodeJS FPS')
                        else:
                            print(f'Level {level_i} not found in nodejs results for game {game}')
                    else:
                        print(f'Game {game} not found in nodejs results')

                    ax.legend()

            rollout_len = get_step_int(rollout_len_str)
            fig.suptitle(f'{device} -- {rollout_len}-step random rollout', fontsize=16)
            fig.tight_layout()
            fig.savefig(f'plots/{device}_{rollout_len_str}{('_select' if not cfg.all_games else '')}.png')

            
if __name__ == '__main__':
    main()