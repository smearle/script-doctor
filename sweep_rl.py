import copy
import hydra
import submitit

from conf.config import TrainConfig
from train import main as main_train


game_to_n_envs = {
    'sokoban_basic': 600,
    'limerick': 300,
}


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: TrainConfig):

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

    executor.map_array(
        main_train,
        sweep_configs,
    )


if __name__ == "__main__":
    main()