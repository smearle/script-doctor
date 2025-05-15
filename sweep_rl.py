import copy
import os
from typing import List
import hydra
import pandas as pd
import submitit

from conf.config import SweepRLConfig, TrainConfig
from train import main as main_train
from utils_rl import init_config


game_to_n_envs = {
    'sokoban_basic': 600,
    'limerick': 300,
}


def plot_rl_runs(cfgs: List[SweepRLConfig]):
    cfgs = [init_config(cfg) for cfg in cfgs]
    for cfg in cfgs:
        exp_dir = cfg._exp_dir
        progress_csv = os.path.join(exp_dir, "progress.csv")
        df = pd.read_csv(progress_csv)
        breakpoint()


@hydra.main(version_base="1.3", config_path="conf", config_name="sweep_rl_config")
def main(cfg: SweepRLConfig):

    seeds = list(range(0, 5))

    cfg.n_envs = game_to_n_envs[cfg.game]

    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
        slurm_job_name=f"{cfg.game}-{cfg.level}",
        mem_gb=30,
        tasks_per_node=1,
        cpus_per_task=1,
        timeout_min=1440,
        slurm_gres='gpu:1',
        slurm_account='pr_174_tandon_advanced', 
    )
    sweep_configs = []
    for seed in seeds:
        cfg_i = copy.deepcopy(cfg)
        cfg_i.seed = seed
        sweep_configs.append(cfg_i)

    if cfg.plot:
        plot_rl_runs(sweep_configs)

    executor.map_array(
        main_train,
        sweep_configs,
    )


if __name__ == "__main__":
    main()