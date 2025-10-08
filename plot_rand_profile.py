import json
import os

import hydra
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from puzzlejax.conf.config import PlotRandProfileConfig
from puzzlejax.globals import PLOTS_DIR, GAMES_TO_N_RULES_PATH, PRIORITY_GAMES
from puzzlejax.preprocess_games import count_rules
from profile_rand_jax import get_step_int, get_level_int, get_vmap, VMAPS
from puzzlejax.globals import STANDALONE_NODEJS_RESULTS_PATH, JAX_PROFILING_RESULTS_DIR
from puzzlejax.utils import init_ps_env


# GAMES_TO_PLOT = [
#     'sokoban_basic',
#     'test_sokoban_rules_3',
#     'test_sokoban_rules_5',

#     # 'sokoban_match3',
#     # 'limerick',
#     # 'blocks',
#     # 'slidings',
#     # 'notsnake',
#     # 'Travelling_salesman',
#     # 'Zen_Puzzle_Garden',
#     # # 'Multi-word_Dictionary_Game',
#     # 'Take_Heart_Lass',
# ]
GAMES_TO_PLOT = PRIORITY_GAMES




@hydra.main(version_base="1.3", config_path="conf", config_name="plot_rand_profile_config")
def main(cfg: PlotRandProfileConfig):
    devices = os.listdir(JAX_PROFILING_RESULTS_DIR)

    with open(STANDALONE_NODEJS_RESULTS_PATH, 'r') as f:
        results_standalone = json.load(f)

    with open(GAMES_TO_N_RULES_PATH, 'r') as f:
        games_to_n_rules = json.load(f)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    for device in devices:
        print(f'Device: {device}')
        rollout_len_strs = os.listdir(os.path.join(JAX_PROFILING_RESULTS_DIR, device))
        for rollout_len_str in rollout_len_strs:
            print(f'Rollout len: {rollout_len_str}')

            if cfg.all_games:
                games = os.listdir(os.path.join(JAX_PROFILING_RESULTS_DIR, device, rollout_len_str))
            else:
                games = GAMES_TO_PLOT

            games_n_rules = []
            for game in games:
                if game+'.txt' not in games_to_n_rules:
                    print(f'Game {game} not found in games_to_n_rules. You may need to run preprocess_games.py first.')
                    env = init_ps_env(game=game, level_i=0, max_episode_steps=1000, vmap=True)
                    n_rules = count_rules(env.tree)
                    games_to_n_rules[game+'.txt'] = (n_rules, env.has_randomness())
                games_n_rules.append((game, games_to_n_rules[game+'.txt']))

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

                levels = os.listdir(os.path.join(JAX_PROFILING_RESULTS_DIR, device, rollout_len_str, game))
                plotted_nodejs = {}
                for level_path in levels:
                    level_str = level_path[:-5]  # Remove .json extension
                    level_i = get_level_int(level_str)
                    if level_i != 0:
                        print(f"Ignoring levels other than 0 for now ({game})")
                        continue
                    vmap = get_vmap(level_str)
                    if vmap not in VMAPS:
                        continue

                    level_results_path = os.path.join(JAX_PROFILING_RESULTS_DIR, device, rollout_len_str, game,
                                                      level_path)
                    with open(level_results_path, 'r') as f:
                        n_envs_to_fps = json.load(f)

                    n_envs = list(n_envs_to_fps.keys())
                    n_envs = [int(n_env) for n_env in n_envs]
                    fpss = list(n_envs_to_fps.values())
                    fps = [f[-1] for f in fpss]
                    sorted_idxs = sorted(range(len(n_envs)), key=lambda k: int(n_envs[k]))
                    n_envs = [n_envs[i] for i in sorted_idxs]
                    fps = [fps[i] for i in sorted_idxs]
                    
                    label = 'PuzzleJAX' if vmap else 'PuzzleJAX (for loop)'
                    color = 'C0' if vmap else 'C1'
                    ax.plot(n_envs, fps, label=label, marker='x', markersize=5, linestyle='-', color=color)

                    # Make the y-axis logarithmic
                    ax.set_yscale('linear')
                    ax.set_xscale('linear')
                    # ax.set_xscale('log')
                    # ax.set_yscale('log')

                    ax.set_xlabel('batch size')
                    ax.set_ylabel('FPS')
                    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) 
                    ax.grid(True)
                    n_rules, has_randomness = games_to_n_rules[game+'.txt']
                    ax.set_title(f'{game}\n({n_rules} rule{"s" if n_rules != 1 else ""}{", stochastic" if has_randomness else ""})')

                    print(f'Game: {game}')

                    # Plot each of the random rollout FPS's from nodejs as broken lines running horizontally
                    run_names = results_standalone.keys()
                    run_name = 'algo-randomRollout_5000-steps_11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz'
                    run_name_python = 'algo-rand_rollout_from_python_5000-steps_11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz'

                    if level_i in plotted_nodejs:
                        continue
                    
                    if game in results_standalone[run_name]:
                        if str(level_i) in results_standalone[run_name][game]:
                            if "Error" in results_standalone[run_name][game][str(level_i)]:
                                print(f'Error in nodejs results for game {game} level {level_i}: {results_standalone[run_name][game][str(level_i)]["Error"]}')
                            else:
                                nodejs_rand_rollout_fps = results_standalone[run_name][game][str(level_i)]['FPS']
                                ax.axhline(y=nodejs_rand_rollout_fps, color='C3', linestyle='--', label='NodeJS')
                        else:
                            print(f'Level {level_i} not found in nodejs results for game {game}')
                    else:
                        print(f'Game {game} not found in nodejs results')
                    if game in results_standalone[run_name_python]:
                        if str(level_i) in results_standalone[run_name_python][game]:
                            if "Error" in results_standalone[run_name_python][game][str(level_i)]:
                                print(f'Error in nodejs results for game {game} level {level_i}: {results_standalone[run_name_python][game][str(level_i)]["Error"]}')
                            else:
                                nodejs_rand_rollout_python_fps = results_standalone[run_name_python][game][str(level_i)]['FPS']
                                ax.axhline(y=nodejs_rand_rollout_python_fps, color='C2', linestyle='--', label='Python-NodeJS')
                        else:
                            print(f'Level {level_i} not found in nodejs results for game {game}')
                    else:
                        print(f'Game {game} not found in nodejs results')

                    plotted_nodejs[level_i] = True

                # if game_i == len(games_n_rules) - 1:
                if True:
                    handles, labels = ax.get_legend_handles_labels()
                    label_to_handle = dict(zip(labels, handles))

                    labels_order = ['PuzzleJAX', 'PuzzleJAX (for loop)', 'NodeJS', 'Python-NodeJS']
                    ordered_handles = [label_to_handle[label] for label in labels_order if label in label_to_handle]
                    ordered_labels = [label for label in labels_order if label in label_to_handle]
                    ax.legend(handles=ordered_handles, labels=ordered_labels)
                    print(ordered_labels)

            rollout_len = get_step_int(rollout_len_str)
            fig.suptitle(f'{device} -- {rollout_len}-step random rollout', fontsize=16)
            fig.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, f'{device}_{rollout_len_str}{("_select" if not cfg.all_games else "")}.png')
            plot_path = plot_path.replace(' ', '_')
            print(f'Saving plot to {plot_path}')
            fig.savefig(plot_path)

            
if __name__ == '__main__':
    main()
