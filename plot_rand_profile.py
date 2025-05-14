import json
import os

from matplotlib import pyplot as plt

from globals import PLOTS_DIR
from profile_rand_jax import JAX_N_ENVS_TO_FPS_PATH, get_step_int, get_level_int


def main():
    with open(JAX_N_ENVS_TO_FPS_PATH, 'r') as f:
        results = json.load(f)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    devices = results.keys()
    for device in devices:
        print(f'Device: {device}')
        rollout_len_str = results[device].keys()
        for rollout_len_str in rollout_len_str:
            print(f'Rollout len: {rollout_len_str}')
            games = results[device][rollout_len_str].keys()

            n_games = len(games)
            fig, axes = plt.subplots(n_games, 1, figsize=(10, 5 * n_games), constrained_layout=True)

            if n_games == 1:
                axes = [axes]

            for game_i, game in enumerate(games):
                ax = axes[game_i]

                levels = results[device][rollout_len_str][game].keys()
                for level_str in levels:
                    level_i = get_level_int(level_str)

                    n_envs_to_fps = results[device][rollout_len_str][game][level_str]

                    n_envs = list(n_envs_to_fps.keys())
                    n_envs = [int(n_env) for n_env in n_envs]
                    fps = list(n_envs_to_fps.values())
                    sorted_idxs = sorted(range(len(n_envs)), key=lambda k: int(n_envs[k]))
                    n_envs = [n_envs[i] for i in sorted_idxs]
                    fps = [fps[i] for i in sorted_idxs]
                    
                    ax.plot(n_envs, fps, label=level_str)

                # Make the y-axis logarithmic
                ax.set_yscale('log')
                ax.set_xlabel('batch size')
                ax.set_ylabel('FPS')
                ax.grid(True)
                ax.set_title(f'{game}')
                ax.legend()

                 
                print(f'Game: {game}')

            rollout_len = get_step_int(rollout_len_str)
            fig.suptitle(f'{device} -- {rollout_len}-step random rollout', fontsize=16)
            fig.savefig(f'plots/{device}_{rollout_len_str}.png')

            
if __name__ == '__main__':
    main()