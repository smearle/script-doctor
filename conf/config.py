from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from hydra.core.config_store import ConfigStore
import numpy as np


@dataclass
class PreprocessConfig:
    game: Optional[str] = None
    overwrite: bool = False

    
@dataclass
class PlotRandProfileConfig:
    all_games: bool = True  # Plot as many games as we have (partial) results for

@dataclass
class PlotStandaloneBFS:
    all_games: bool = True
    aggregate: bool = False  # (Re-)collect all the solution JSONs to compile a results dict for plotting

@dataclass
class PSConfig:
    game: str = "sokoban_basic"
    level: int = 0
    max_episode_steps: int = np.iinfo(np.int32).max
    overwrite: bool = False  # Whether to overwrite existing results


@dataclass
class JaxValidationConfig(PSConfig):
    all_games: bool = True
    slurm: bool = False
    n_games_per_job: int = 1
    game: Optional[str] = None
    aggregate: bool = False  # Don't run any new validations, just aggregate existing results.
    random_order: bool = False
    include_test_games: bool = True
    
@dataclass
class BFSConfig(PSConfig):
    game: Optional[str] = None
    max_steps: int = 100_000
    n_best_to_keep: int = 1
    render_gif: bool = True
    render_live: bool = False
    all_games: bool = True


@dataclass
class ProfileJaxRandConfig(PSConfig):
    game: Optional[str] = None
    all_games: bool = False
    n_steps: int = 5_000
    # reevaluate: bool = True  # Whether to continue profiling, or just plot the results
    render: bool = False


    
@dataclass
class ProfileNodeJS(PSConfig):
    # algo: str = "random"  # 'bfs', 'random'
    algo: str = "bfs"  # 'bfs', 'random'
    game: Optional[str] = None
    # all_games: bool = False
    all_games: bool = True
    random_order: bool = False
    # n_steps: int = 5_000
    n_steps: int = 1_000_000
    overwrite: bool = False
    include_randomness: bool = True
    # timeout: int = 60
    timeout: int = -1
    for_validation: bool = False
    # for_validation: bool = True
    for_solution: bool = True
    slurm: bool = False
    n_games_per_job: int = 1


@dataclass
class RLConfig(PSConfig):
    max_episode_steps: int = 100
    lr: float = 1.0e-4
    n_envs: int = 400
    # How many steps do I take in all of my batched environments before doing a gradient update
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    timestep_chunk_size: int = -1
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "PSEnv"
    ANNEAL_LR: bool = False
    DEBUG: bool = True
    exp_name: str = "0"
    seed: int = 0

    model: str = "conv2"

    map_width: int = 16
    randomize_map_shape: bool = False
    is_3d: bool = False
    # ctrl_metrics: Tuple[str] = ('diameter', 'n_regions')
    ctrl_metrics: Tuple[str] = ()
    # Size of the receptive field to be fed to the action subnetwork.
    vrf_size: Optional[int] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # Size of the receptive field to be fed to the value subnetwork.
    arf_size: Optional[int] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # TODO: actually take arf and vrf into account in models, where possible

    change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.
    act_shape: Tuple[int, int] = (1, 1)

    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1  # multi-agent is fake and broken
    multiagent: bool = False

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: float = 0.1

    # To make the task simpler, always start with an empty map
    empty_start: bool = False
    # Or a full (all-wall) map
    full_start: bool = False

    # In problems with tile-types with specified valid numbers, fix/freeze their random placement at the beginning of 
    # each episode.
    pinpoints: bool = False

    hidden_dims: Tuple[int] = (128, 128)

    # TODO: Implement this. Just a placeholder for now.
    reward_every: int = 1

    # A toggle, will add `n_envs` to the experiment name if we are profiling training FPS, so that we can distinguish 
    # results.
    profile_fps: bool = False

    reward_freq: int = 1

    overwrite: bool = False


    """ DO NOT USE. WILL BE OVERWRITTEN. """
    _exp_dir: Optional[str] = None
    _n_gpus: int = 1


@dataclass
class TrainConfig(RLConfig):
    overwrite: bool = False

    # WandB Params
    wandb_mode: str = 'run'  # one of: 'offline', 'run', 'dryrun', 'shared', 'disabled', 'online'
    wandb_entity: str = ''
    wandb_project: str = 'smearle_ps_ppo'

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1e7)
    # Render after this many update steps
    render_freq: int = 50
    n_render_eps: int = 3


    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    _num_updates: Optional[int] = None
    _minibatch_size: Optional[int] = None
    _ckpt_dir: Optional[str] = None
    _is_recurrent: bool = False
    _exp_dir: Optional[str] = None
    _vid_dir: Optional[str] = None
    ###########################################################################


@dataclass
class SweepRLConfig(TrainConfig):
    game: Optional[str] = None
    all_games: bool = False
    plot: bool = False
    slurm: bool = True


cs = ConfigStore.instance()
cs.store(name="preprocess_config", node=PreprocessConfig)
cs.store(name="plot_rand_profile_config", node=PlotRandProfileConfig)
cs.store(name="plot_standalone_bfs_config", node=PlotStandaloneBFS)
cs.store(name="ps_config", node=PSConfig)
cs.store(name="jax_validation_config", node=JaxValidationConfig)
cs.store(name="config", node=RLConfig)
cs.store(name="bfs_config", node=BFSConfig)
cs.store(name="profile_jax_config", node=ProfileJaxRandConfig)
cs.store(name="profile_nodejs_config", node=ProfileNodeJS)
cs.store(name="train_config", node=TrainConfig)
cs.store(name="sweep_rl_config", node=SweepRLConfig)
