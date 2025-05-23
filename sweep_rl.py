from collections import defaultdict
import copy
import glob
import json
import os
from typing import List
import hydra
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import submitit

from conf.config import SweepRLConfig, TrainConfig
from env import PSParams
from preprocess_games import get_env_from_ps_file
from train import main as main_train
from utils import get_list_of_games_for_testing, get_n_levels_per_game, init_ps_lark_parser
from utils_rl import get_env_params_from_config, init_config
from validate_sols import JS_TO_JAX_ACTIONS, JS_SOLS_DIR, SOLUTION_REWARDS_PATH


def replay_solution(parser, game, level_i, actions):
    actions = [JS_TO_JAX_ACTIONS[a] for a in actions]
    actions = jnp.array([int(a) for a in actions], dtype=jnp.int32)
    env, tree, success, err_msg = get_env_from_ps_file(parser, game)
    key = jax.random.PRNGKey(0)
    level = env.get_level(level_i)
    params = PSParams(
        level=level
    )
    obs, state = env.reset(key, params)
    def step_env(state, action):
        obs, state, reward, done, info = env.step_env(key, state, action, params)
        return state, (state, reward)
    state, (state_v, reward_v) = jax.lax.scan(step_env, state, actions)
    reward = float(reward_v.sum().item())
    return reward


def plot_rl_runs_reward(grid_cfgs: List[SweepRLConfig]):
    # Group configs by game
    if os.path.isfile(SOLUTION_REWARDS_PATH):
        with open(SOLUTION_REWARDS_PATH, 'r') as f:
            solution_rewards = json.load(f)
    else:
        solution_rewards = {}
    parser = init_ps_lark_parser()
    for game in grid_cfgs:
        game_cfgs = grid_cfgs[game]
        n_levels = len(game_cfgs)
        # fig, axs = plt.subplots(n_levels, 1, figsize=(8, 4 * n_levels), sharex=True)
        # if n_levels == 1:
        #     axs = [axs]
        for level_i, level in enumerate(game_cfgs):
            # ax = axs[level_i]

            level_cfgs = game_cfgs[level]
            dfs = []
            for cfg in [init_config(cfg) for cfg in level_cfgs]:
                exp_dir = cfg._exp_dir
                progress_csv = os.path.join(exp_dir, "progress.csv")
                if not os.path.isfile(progress_csv):
                    print(f"{progress_csv} does not exist, skipping.")
                    continue
                df = pd.read_csv(progress_csv)[['timestep', 'ep_return']]
                df = df[['timestep', 'ep_return']].drop_duplicates(subset='timestep')
                # Replace NaNs with 0s
                dfs.append(df)

            if not dfs:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))

            # Interpolate all dfs on a shared set of timesteps
            common_timesteps = sorted(set().union(*[df['timestep'].values for df in dfs]))
            reindexed_dfs = [df.set_index('timestep').reindex(common_timesteps).interpolate() for df in dfs]
            # reindexed_dfs = [df.fillna(0) for df in dfs]

            # Here we assume these runs all contain the same timestep labels
            # common_timesteps = df['timestep'].values
            # reindexed_dfs = dfs

            stacked = pd.concat(reindexed_dfs, axis=1)
            ep_returns = stacked.filter(like='ep_return')

            mean = ep_returns.mean(axis=1)
            std = ep_returns.std(axis=1)

            ax.plot(common_timesteps, mean, label='Mean Return')
            ax.fill_between(common_timesteps, mean - std, mean + std, alpha=0.3, label='Std Dev')
            ax.set_title(f"{game}, level {level}")
            ax.set_ylabel("Episodic Return")
            ax.set_xlabel("Timesteps")

            level_sols = glob.glob(os.path.join(JS_SOLS_DIR, game, f"*level-{level}.json"))
            n_search_steps = [level_sols.split('-steps')[0].split('/')[-1] if '-steps' in level_sols else 100_000 for level_sols in level_sols]
            sorted_idxs = np.argsort(n_search_steps)
            level_sols = [level_sols[i] for i in sorted_idxs]
            n_search_steps = [int(n_search_steps[i]) for i in sorted_idxs]
            n_search_steps_to_colors = {
                100_000: 'purple',
                1_000_000: 'red',
            }
            for n_search_steps, level_sol in zip(n_search_steps, level_sols):
                if str((game, level, n_search_steps)) not in solution_rewards:
                    with open(level_sol, 'r') as f:
                        level_sol_dict = json.load(f)
                    solved = level_sol_dict['won']
                    reward = replay_solution(parser, game, level, level_sol_dict['actions'])
                    solution_rewards[str((game, level, n_search_steps))] = {
                        'solved': solved,
                        'reward': reward,
                    }
                    with open(SOLUTION_REWARDS_PATH, 'w') as f:
                        json.dump(solution_rewards, f)
                else:
                    solved = solution_rewards[str((game, level, n_search_steps))]['solved']
                    reward = solution_rewards[str((game, level, n_search_steps))]['reward']
                # Now plot the solution reward as a broken horizontal line on the same plot (yellow for 100k steps, red
                # for 1M steps, with a label indicating the number of steps and either solved or unsolved)
                color = n_search_steps_to_colors.get(n_search_steps, 'black')
                solved_label = 'Solved' if solved else 'Unsolved'
                ax.axhline(y=reward, color=color, linestyle='--', label=f"BFS {n_search_steps:,} steps ({solved_label})")
                if solved:
                    break

            ax.legend()
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            save_path = os.path.join('plots', f"{game}_level-{level}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved plot for game '{game}' level {level} to {save_path}.")


        # axs[-1].set_xlabel("Timesteps")
        # fig.suptitle(f"Game: {game}")

def gen_grid_cfgs(sweep_cfg: SweepRLConfig):
    """Generate a dictionary mapping games to levels to lists of RL training configs."""
    games_to_n_levels = get_n_levels_per_game()
    if sweep_cfg.game is None:
        games = get_list_of_games_for_testing(all_games=sweep_cfg.all_games)
    else:
        games = [sweep_cfg.game]
    grid_cfgs = {}
    for game in games:
        grid_cfgs[game] = {}
        if game in GAME_TO_N_ENVS:
            n_envs = GAME_TO_N_ENVS[game]
        else:
            n_envs = sweep_cfg.n_envs
        n_levels = games_to_n_levels[game]
        for level in range(n_levels):
            exp_cfgs = []
            for seed in SEEDS:
                cfg_i = TrainConfig()
                cfg_i.seed = seed
                cfg_i.game = game
                cfg_i.level = level
                cfg_i.n_envs = n_envs
                cfg_i.total_timesteps = int(TOTAL_TIMESTEPS)
                exp_cfgs.append(cfg_i)
            grid_cfgs[game][level] = exp_cfgs
    return grid_cfgs


SEEDS = list(range(10, 15))
# SEEDS = [0]

GAME_TO_N_ENVS = {
    'sokoban_basic': 100,
    'limerick': 300,
}

# TOTAL_TIMESTEPS = 1e6
TOTAL_TIMESTEPS = 5e7


@hydra.main(version_base="1.3", config_path="conf", config_name="sweep_rl_config")
def main(sweep_cfg: SweepRLConfig):

    grid_cfgs = gen_grid_cfgs(sweep_cfg)
    if sweep_cfg.plot:
        return plot_rl_runs_reward(grid_cfgs)

    for game in grid_cfgs:
        game_cfgs = grid_cfgs[game]
        for level in game_cfgs:
            level_cfgs = game_cfgs[level]
            seeds = [cfg.seed for cfg in level_cfgs]
            n_envs = level_cfgs[0].n_envs
            print(f"Launching jobs {seeds} for {game} level {level}, with {n_envs}.")

            level_cfgs = [OmegaConf.create(c) for c in level_cfgs]
            if sweep_cfg.slurm:
                executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "rl"))
                executor.update_parameters(
                    slurm_job_name=f"{game}-{level}",
                    mem_gb=30,
                    tasks_per_node=1,
                    cpus_per_task=1,
                    timeout_min=1440,
                    slurm_gres='gpu:1',
                    slurm_account='pr_174_tandon_advanced', 
                )

                executor.map_array(
                    main_train,
                    level_cfgs,
                )

            else:
                [main_train(cfg) for cfg in level_cfgs]

if __name__ == "__main__":
    main()