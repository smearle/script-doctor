import copy
import glob
import json
import os
from dataclasses import fields
from itertools import product
from typing import Any, Dict, List, Sequence, Tuple

import dotenv
import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import ListConfig, OmegaConf
import pandas as pd
import submitit

from puzzlejax.conf.config import SweepRLConfig, TrainConfig, EnjoyConfig
from puzzlejax.env import PJParams
from puzzlejax.preprocess_games import get_env_from_ps_file
from train import main as main_train
from enjoy import main_enjoy
from sweep_rl_configs import _NAMED_SWEEPS
from puzzlejax.utils import get_list_of_games_for_testing, get_n_levels_per_game, init_ps_lark_parser
from utils_rl import init_config
from puzzlejax.globals import JS_TO_JAX_ACTIONS, JS_SOLS_DIR, SOLUTION_REWARDS_PATH


dotenv.load_dotenv()


SWEEP_META_FIELDS = {
    "game",
    "all_games",
    "plot",
    "success_heatmap",
    "slurm",
    "mode",
    "render_ims",
    "sweep_name",
    "sweep_axes",
}


def replay_solution(parser, game, level_i, actions):
    actions = [JS_TO_JAX_ACTIONS[a] for a in actions]
    actions = jnp.array([int(a) for a in actions], dtype=jnp.int32)
    env, tree, success, err_msg = get_env_from_ps_file(parser, game)
    key = jax.random.PRNGKey(0)
    level = env.get_level(level_i)
    params = PJParams(
        level=level
    )
    obs, state = env.reset(key, params)
    def step_env(state, action):
        obs, state, reward, done, info = env.step_env(key, state, action, params)
        return state, (state, reward)
    state, (state_v, reward_v) = jax.lax.scan(step_env, state, actions)
    reward = float(reward_v.sum().item())
    return reward


def _run_field_names(cfg_cls) -> set[str]:
    return {f.name for f in fields(cfg_cls)}


def _cfg_items(cfg_obj: Any):
    if hasattr(cfg_obj, "items"):
        return cfg_obj.items()
    return vars(cfg_obj).items()


def _canonical_value(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_canonical_value(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_canonical_value(x) for x in v)
    return v


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2e}" if (abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e3)) else f"{v:g}"
    return str(v)


def _infer_varying_fields(cfgs: Sequence[TrainConfig], candidate_fields: Sequence[str] | None = None) -> List[str]:
    if not cfgs:
        return []
    if candidate_fields is None:
        candidate_fields = sorted(_run_field_names(type(cfgs[0])))
    varying = []
    for field_name in candidate_fields:
        vals = {_canonical_value(getattr(cfg, field_name)) for cfg in cfgs}
        if len(vals) > 1:
            varying.append(field_name)
    return varying


def _curve_key(cfg: TrainConfig, curve_fields: Sequence[str]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((f, _canonical_value(getattr(cfg, f))) for f in curve_fields)


def _curve_label(curve_key: Tuple[Tuple[str, Any], ...]) -> str:
    if not curve_key:
        return "mean"
    return ", ".join(f"{k}={_format_value(v)}" for k, v in curve_key)


def _extract_sweep_axes(sweep_cfg: SweepRLConfig, run_fields: set[str]) -> Dict[str, List[Any]]:
    raw_axes = getattr(sweep_cfg, "sweep_axes", {}) or {}
    sweep_axes: Dict[str, List[Any]] = {}
    for field_name, values in raw_axes.items():
        if field_name not in run_fields:
            raise ValueError(f"sweep_axes contains unknown run field {field_name!r}.")
        if isinstance(values, (list, tuple, ListConfig)):
            values_list = list(values)
        else:
            values_list = [values]
        sweep_axes[field_name] = values_list

    return sweep_axes


def _expand_sweep_updates(sweep_axes: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not sweep_axes:
        return [{}]
    if any(len(v) == 0 for v in sweep_axes.values()):
        return []
    axis_names = list(sweep_axes.keys())
    axis_values = [sweep_axes[name] for name in axis_names]
    return [
        {name: value for name, value in zip(axis_names, combo)}
        for combo in product(*axis_values)
    ]


def _coerce_to_field_type(cfg_obj: Any, field_name: str, value: Any) -> Any:
    """Coerce sweep values to match existing config field container types."""
    current = getattr(cfg_obj, field_name, None)
    if isinstance(current, tuple):
        if isinstance(value, (list, tuple, ListConfig)):
            return tuple(value)
        return (value,)
    return value


def plot_rl_runs_reward(grid_cfgs: List[SweepRLConfig]):
    # wandb_api = wandb.Api()
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
            varying_fields = _infer_varying_fields(level_cfgs)
            curve_fields = [f for f in varying_fields if f != "seed"]
            curve_to_dfs: Dict[Tuple[Tuple[str, Any], ...], List[pd.DataFrame]] = {}
            for cfg in [init_config(cfg) for cfg in level_cfgs]:
                exp_dir = cfg._exp_dir
                progress_csv = os.path.join(exp_dir, "progress.csv")
                if not os.path.isfile(progress_csv):
                    print(f"{progress_csv} does not exist, skipping.")
                    continue
                df = pd.read_csv(progress_csv)[['timestep', 'ep_return']]
                df = df[['timestep', 'ep_return']].drop_duplicates(subset='timestep')
                # Replace NaNs with 0s
                if len(df) == 0:
                    print(f"{progress_csv} is empty, skipping.")
                    continue
                curve_to_dfs.setdefault(_curve_key(cfg, curve_fields), []).append(df)

                # Load wandb history
                # print(f"Loading wandb run for {cfg._exp_dir}...")
                # wandb_path = os.path.join(exp_dir, 'wandb_run_id.txt')
                # with open(wandb_path, 'r') as f:
                #     wandb_run_id = f.read()
                # try:
                #     sc_run = wandb_api.run(f'/{cfg.wandb_project}/{wandb_run_id}')
                # except wandb.errors.CommError:
                #     wandb_run_dirs = os.listdir(os.path.join(exp_dir, 'wandb'))
                #     for d in wandb_run_dirs:
                #         if d.startswith(f'run-{wandb_run_id}'):
                #             os.system(f'wandb sync {os.path.join(exp_dir, "wandb", d)}')
                # train_metrics = sc_run.history()

            if not curve_to_dfs:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))
            for key, dfs in sorted(curve_to_dfs.items(), key=lambda x: str(x[0])):
                # Interpolate all dfs on a shared set of timesteps
                common_timesteps = sorted(set().union(*[df['timestep'].values for df in dfs]))
                common_timesteps = [t for t in common_timesteps if t is not None]
                reindexed_dfs = [df.set_index('timestep').reindex(common_timesteps).interpolate() for df in dfs]

                stacked = pd.concat(reindexed_dfs, axis=1)
                ep_returns = stacked.filter(like='ep_return')

                mean = ep_returns.mean(axis=1)
                std = ep_returns.std(axis=1)
                label = _curve_label(key)

                ax.plot(common_timesteps, mean, label=f"{label} mean")
                ax.fill_between(common_timesteps, mean - std, mean + std, alpha=0.2, label=f"{label} std")
            ax.set_title(f"{game}, level {level}")
            ax.set_ylabel("Episodic Return")
            ax.set_xlabel("Timesteps")

            level_sols = glob.glob(os.path.join(JS_SOLS_DIR, game, f"*level-{level}.json"))
            print(level_sols)
            n_search_steps = [level_sols.split('-steps')[0].split('/')[-1].split('_')[1] if '-steps' in level_sols else 100_000 for level_sols in level_sols]
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

def gen_grid_cfgs(sweep_cfg: SweepRLConfig, base_cfg: TrainConfig):
    """Generate a dictionary mapping games to levels to lists of RL training configs."""
    run_fields = _run_field_names(type(base_cfg))
    sweep_axes = _extract_sweep_axes(sweep_cfg, run_fields)
    sweep_updates = _expand_sweep_updates(sweep_axes)
    sweep_axis_names = set(sweep_axes.keys())

    games_to_n_levels = get_n_levels_per_game()
    if sweep_cfg.game is None:
        games = get_list_of_games_for_testing(all_games=sweep_cfg.all_games)
    else:
        games = [sweep_cfg.game]
    grid_cfgs = {}
    for game in games:
        grid_cfgs[game] = {}
        if game in GAME_TO_N_ENVS:
            game_default_n_envs = GAME_TO_N_ENVS[game]
        else:
            game_default_n_envs = sweep_cfg.n_envs
        n_levels = games_to_n_levels[game]
        for level in range(n_levels):
            exp_cfgs = []
            for updates in sweep_updates:
                cfg_i = copy.deepcopy(base_cfg)
                for k, v in _cfg_items(sweep_cfg):
                    if k in SWEEP_META_FIELDS:
                        continue
                    if k in sweep_axis_names:
                        continue
                    if hasattr(cfg_i, k):
                        setattr(cfg_i, k, _coerce_to_field_type(cfg_i, k, v))
                for k, v in updates.items():
                    setattr(cfg_i, k, _coerce_to_field_type(cfg_i, k, v))
                cfg_i.game = game
                cfg_i.level = level
                if "n_envs" not in updates:
                    cfg_i.n_envs = game_default_n_envs
                cfg_i.total_timesteps = int(TOTAL_TIMESTEPS)
                exp_cfgs.append(cfg_i)
            grid_cfgs[game][level] = exp_cfgs
    return grid_cfgs


GAME_TO_N_ENVS = {
#     'sokoban_basic': 100,
#     'limerick': 300,
}

# TOTAL_TIMESTEPS = 1e6
TOTAL_TIMESTEPS = 5e7


def _apply_named_sweep(sweep_cfg: SweepRLConfig) -> SweepRLConfig:
    """Apply a named sweep preset unless keys were explicitly overridden via Hydra."""
    sweep_name = str(getattr(sweep_cfg, "sweep_name", "sweep"))
    preset_cls = _NAMED_SWEEPS.get(sweep_name)
    if preset_cls is None:
        known = ", ".join(sorted(_NAMED_SWEEPS.keys()))
        raise ValueError(f"Unknown sweep_name={sweep_name!r}. Known: {known}")

    preset = preset_cls()
    overridden_root_keys = set()
    overridden_sweep_axes_keys = set()
    sweep_axes_overridden = False
    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        overrides = []
    for override in overrides:
        if "=" not in override:
            continue
        key = override.split("=", 1)[0].lstrip("+")
        overridden_root_keys.add(key.split(".", 1)[0])
        if key == "sweep_axes":
            sweep_axes_overridden = True
        elif key.startswith("sweep_axes."):
            overridden_sweep_axes_keys.add(key.split(".", 1)[1])

    resolved_cfg = copy.deepcopy(sweep_cfg)
    if "sweep_axes" not in overridden_root_keys and not sweep_axes_overridden:
        setattr(resolved_cfg, "sweep_axes", copy.deepcopy(preset.sweep_axes))
    elif not sweep_axes_overridden:
        merged_sweep_axes = dict(getattr(resolved_cfg, "sweep_axes", {}) or {})
        for axis_name, axis_values in preset.sweep_axes.items():
            if axis_name not in overridden_sweep_axes_keys:
                merged_sweep_axes[axis_name] = axis_values
        setattr(resolved_cfg, "sweep_axes", merged_sweep_axes)
    return resolved_cfg


@hydra.main(version_base="1.3", config_path="puzzlejax/conf", config_name="sweep_rl_config")
def main(sweep_cfg: SweepRLConfig):
    sweep_cfg = _apply_named_sweep(sweep_cfg)
    if sweep_cfg.mode == 'train':
        main_fn = main_train
        CfgCls = TrainConfig
    elif sweep_cfg.mode == 'enjoy':
        main_fn = main_enjoy
        CfgCls = EnjoyConfig
    base_cfg = CfgCls()

    grid_cfgs = gen_grid_cfgs(sweep_cfg, base_cfg)
    if sweep_cfg.plot:
        return plot_rl_runs_reward(grid_cfgs)

    executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "rl"))
    executor.update_parameters(
        # slurm_job_name=f"{game}-{level}",
        slurm_job_name=f"puzzlejax-ppo",
        mem_gb=30,
        tasks_per_node=1,
        cpus_per_task=1,
        timeout_min=60*24,
        slurm_gres='gpu:1',
        slurm_array_parallelism=1_000,
        slurm_account=os.environ.get("SLURM_ACCOUNT")
    )
    all_cfgs = []

    for game in grid_cfgs:
        game_cfgs = grid_cfgs[game]
        for level in game_cfgs:
            level_cfgs = game_cfgs[level]
            if not level_cfgs:
                print(f"No runs scheduled for {game} level {level} (empty sweep axis).")
                continue
            varying_fields = _infer_varying_fields(level_cfgs)
            sweep_summary = {
                field: sorted({_canonical_value(getattr(cfg, field)) for cfg in level_cfgs}, key=str)
                for field in varying_fields
            }
            n_envs = level_cfgs[0].n_envs
            print(
                f"Launching {len(level_cfgs)} jobs for {game} level {level}: "
                f"sweep={sweep_summary}, n_envs={n_envs}."
            )

            level_cfgs = [OmegaConf.create(c) for c in level_cfgs]
            all_cfgs.append(level_cfgs)

    all_cfgs = [cfg for sublist in all_cfgs for cfg in sublist]  # Flatten list of lists
    if not all_cfgs:
        print("No runs scheduled (empty sweep axes).")
        return
    all_cfgs = sorted(all_cfgs, key=lambda x: str(sorted(x.items())))
    if not sweep_cfg.slurm:
        [main_fn(cfg) for cfg in all_cfgs]
    else:
        executor.map_array(
            main_fn,
            all_cfgs,
        )

if __name__ == "__main__":
    main()
