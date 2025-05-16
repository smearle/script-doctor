import copy
import os
from typing import List
import hydra
from omegaconf import OmegaConf
import pandas as pd
import submitit

from conf.config import SweepRLConfig, TrainConfig
from train import main as main_train
from utils import get_list_of_games_for_testing, get_n_levels_per_game
from utils_rl import init_config



def plot_rl_runs(grid_cfgs: List[SweepRLConfig]):
    for game in grid_cfgs:
        game_cfgs = grid_cfgs[game]
        for level in game_cfgs:
            level_cfgs = game_cfgs[level]
            level_cfgs = [init_config(cfg) for cfg in level_cfgs]
            for cfg in level_cfgs:
                exp_dir = cfg._exp_dir
                progress_csv = os.path.join(exp_dir, "progress.csv")
                if not os.path.isfile(progress_csv):
                    print(f"{progress_csv} does not exist, skipping.")
                    continue
                df = pd.read_csv(progress_csv)
                breakpoint()


def gen_grid_cfgs(cfg: SweepRLConfig):
    """Generate a dictionary mapping games to levels to lists of RL training configs."""
    games = get_list_of_games_for_testing(all_games=cfg.all_games)
    games_to_n_levels = get_n_levels_per_game()
    grid_cfgs = {}
    for game in games:
        grid_cfgs[game] = {}
        if game in GAME_TO_N_ENVS:
            n_envs = GAME_TO_N_ENVS[game]
        n_levels = games_to_n_levels[game]
        for level in range(n_levels):
            exp_cfgs = []
            for seed in SEEDS:
                cfg_i = TrainConfig()
                cfg_i.seed = seed
                cfg_i.game = game
                cfg_i.level = level
                cfg.n_envs = n_envs
                exp_cfgs.append(cfg_i)
            grid_cfgs[game][level] = exp_cfgs
    return grid_cfgs

SEEDS = list(range(0, 5))

GAME_TO_N_ENVS = {
    'sokoban_basic': 100,
    'limerick': 300,
}


@hydra.main(version_base="1.3", config_path="conf", config_name="sweep_rl_config")
def main(sweep_cfg: SweepRLConfig):

    grid_cfgs = gen_grid_cfgs(sweep_cfg)
    if sweep_cfg.plot:
        return plot_rl_runs(grid_cfgs)

    for game in grid_cfgs:
        game_cfgs = grid_cfgs[game]
        for level in game_cfgs:
            level_cfgs = game_cfgs[level]
            seeds = [cfg.seed for cfg in level_cfgs]
            n_envs = level_cfgs[0].n_envs
            print(f"Launching jobs {seeds} for {game} level {level}, with {n_envs}.")

            level_cfgs = [OmegaConf.create(c) for c in level_cfgs]
            if sweep_cfg.slurm:
                executor = submitit.AutoExecutor(folder="submitit_logs")
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