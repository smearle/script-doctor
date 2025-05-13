from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

from hydra.core.config_store import ConfigStore
import numpy as np


# @dataclass
# class EnvConfig:
#     problem: str = "binary"
#     representation: str = "nca"


@dataclass
class PSConfig:
    game: str = "sokoban_basic"
    level_i: int = 0
    max_episode_steps: int = np.iinfo(np.int32).max

    
@dataclass
class BFSConfig(PSConfig):
    game: Optional[str] = None
    max_steps: int = 100_000
    n_best_to_keep: int = 1
    render_gif: bool = True
    render_live: bool = False
    all_games: bool = True

    
@dataclass
class ProfileStandalone(PSConfig):
    game: Optional[str] = None
    n_steps: int = 20_000
    overwrite: bool = False


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

    model: str = "conv"

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

    hidden_dims: Tuple[int] = (64, 256)

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
class EvoMapConfig(RLConfig):
    n_generations: int = 100_000
    evo_pop_size: int = 100
    n_parents: int = 50
    mut_rate: float = 0.3
    render_freq: int = 10_000
    log_freq: int = 1_000
    callbacks: bool = True


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
    render_freq: int = 1000
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    _num_updates: Optional[int] = None
    _minibatch_size: Optional[int] = None
    _ckpt_dir: Optional[str] = None
    _is_recurrent: bool = False
    _exp_dir: Optional[str] = None
    _vid_dir: Optional[str] = None
    ###########################################################################


@dataclass
class MultiAgentConfig(TrainConfig):
    multiagent: bool = False
    # lr: float = 3e-4
    # update_epochs: int = 4
    # num_steps: int = 521
    # gamma: float = 0.99
    # gae_lambda: float = 0.95
    # clip_eps: float = 0.2
    # scale_clip_eps: bool = False
    # ent_coef: float = 0.0
    # vf_coef: float = 0.5
    # max_grad_norm: float = 0.25

    model: str = 'rnn'
    n_agents: int = 1
    n_envs: int = 4
    n_eval_envs: int = 10
    scale_clip_eps: bool = False
    hidden_dims: Tuple[int] = (512, -1)

    # Save a checkpoint after (at least) this many ***update*** steps
    ckpt_freq: int = 40
    render_freq: int = 20

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    _num_actors: int = -1
    _num_eval_actors: int = -1
    _minibatch_size: int = -1
    _num_updates: int = -1
    _exp_dir: Optional[str] = None
    _vid_dir: Optional[str] = None
    ###########################################################################


@dataclass
class TrainAccelConfig(TrainConfig):
    evo_freq: int = 10
    evo_pop_size: int = 10
    evo_mutate_prob: float = 0.1


@dataclass
class EvalConfig(TrainConfig):
    reevaluate: bool = True
    random_agent: bool = False
    # In how many bins to divide up each metric being evaluated
    n_bins: int = 10
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[float] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    # Which eval metric to keep in our generated table if sweeping over eval hyperparams (in which case we want to 
    # save space). Only applied when running `cross_eval.py`
    # metrics_to_keep = [
    #     # 'min_min_loss',
    #     'mean_ep_reward',
    # ]

    # metrics_to_keep: Tuple[str] = ('mean_ep_reward',)
    metrics_to_keep: Tuple[str] = ('mean_fps',)


@dataclass
class MultiAgentEvalConfig(EvalConfig, MultiAgentConfig):
    multiagent = True
    pass


@dataclass
class EnjoyConfig(EvalConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    # Add debugging text showing the current/target values for various stats to each frame of the episode (this is really slow)
    render_stats: bool = False
    n_enjoy_envs: int = 1
    render_ims: bool = False
    a_freezer: bool = False


@dataclass
class EnjoyRLConfig(MultiAgentConfig, EnjoyConfig):
    pass
    

@dataclass
class ProfileEnvConfig(RLConfig):
    all_games: bool = True
    n_profile_steps: int = 5000
    reevaluate: bool = True  # Whether to continue profiling, or just plot the results
    render: bool = False


@dataclass
class SweepConfig(EnjoyConfig, EvalConfig):
    name: Optional[str] = None
    mode: str = 'train'
    slurm: bool = True
    overwrite: bool = False

@dataclass
class GetTracesConfig(EnjoyConfig):
    len_trace: int = 3 # deprecated, will filter the dataset later, just leave it here in case
    n_enjoy_envs: int = 1
    n_eps: int = 1  # keep it at 1 for each trace
    render_stats: bool = False
    render_ims: bool = False

cs = ConfigStore.instance()
cs.store(name="config", node=RLConfig)
cs.store(name="bfs_config", node=BFSConfig)
cs.store(name="ma_config", node=MultiAgentConfig)
cs.store(name="enjoy_ma_pcgrl", node=EnjoyRLConfig)
cs.store(name="get_traces_pcgrl", node=GetTracesConfig)
cs.store(name="evo_map_pcgrl", node=EvoMapConfig)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="train_accel_pcgrl", node=TrainAccelConfig)
cs.store(name="enjoy_pcgrl", node=EnjoyConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
# cs.store(name="eval_ma_pcgrl", node=MultiAgentEvalConfig)
cs.store(name="profile_ps", node=ProfileEnvConfig)
cs.store(name="profile_standalone", node=ProfileStandalone)
cs.store(name="batch_pcgrl", node=SweepConfig)
