from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore

from puzzlescript_jax.config import PSConfig


@dataclass
class PreprocessConfig:
    game: Optional[str] = None
    overwrite: bool = False

    
@dataclass
class PlotRandProfileConfig:
    all_games: bool = True  # Plot as many games as we have (partial) results for


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
class CppValidationConfig(PSConfig):
    all_games: bool = True
    slurm: bool = False
    n_games_per_job: int = 1
    game: Optional[str] = None
    aggregate: bool = False
    random_order: bool = False
    include_test_games: bool = True
    render: bool = True
    render_scale: int = 1
    render_mismatches_only: bool = False
    output_dir: str = "data/cpp_validated_js_sols"
    
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
    random_order: bool = False
    # max_episode_steps: int = 100
    n_steps: int = 5_000
    # reevaluate: bool = True  # Whether to continue profiling, or just plot the results
    render: bool = False
    use_switch_env: bool = False  # Use jax.lax.switch-based env for faster compilation
    slurm: bool = False
    n_games_per_job: int = 1


    
@dataclass
class NodeJSConfig(PSConfig):
    game: Optional[str] = None
    level: Optional[int] = None
    all_games: bool = False
    random_order: bool = False
    n_steps: int = 5_000
    overwrite: bool = False
    include_randomness: bool = True
    timeout: int = -1
    render: bool = False
    slurm: bool = False
    n_games_per_job: int = 1


@dataclass
class SearchNodeJSConfig(NodeJSConfig):
    algo: str = "bfs"  # 'bfs', 'astar', 'gbfs', 'mcts', 'random'
    n_steps: int = 100_000
    render: bool = True


@dataclass
class SearchCppConfig(NodeJSConfig):
    algo: str = "bfs"  # 'bfs', 'astar', 'gbfs', 'mcts', 'random'
    n_steps: int = 100_000


@dataclass
class ProfileRandNodeJSConfig(NodeJSConfig):
    pass


@dataclass
class ProfileRandCppConfig(NodeJSConfig):
    cpp_batched_thread_candidates: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32)


@dataclass
class EvolveLevelConfig:
    game: Optional[str] = None
    level: int = 0
    gens: int = 10_000
    pop: int = 6
    n_mutations_min: int = 1
    n_mutations_max: int = 3
    max_nodes: int = 1_000_000
    batch_size: int = 10_000
    cost_weight: float = 0.6
    render_gif: bool = True
    seed: int = 42
    fitness: str = "states"
    allow_player_change: bool = False
    depth_increase_threshold: float = 0.95


@dataclass
class EvolveLevelNodeJSConfig:
    game: Optional[str] = None
    level: int = 0
    gens: int = 10_000
    pop: int = 6
    n_mutations_min: int = 1
    n_mutations_max: int = 3
    max_steps: int = 1_000_000
    timeout: int = -1
    algo: str = "astar"
    seed: int = 42
    fitness: str = "states"
    allow_player_change: bool = False
    render_gif: bool = True
    gif_frame_duration: float = 0.05
    gif_scale: int = 10
    depth_increase_threshold: float = 0.95


@dataclass
class EvolveLevelCppConfig:
    game: Optional[str] = None
    level: int = 0
    gens: int = 10_000
    pop: int = 6
    n_mutations_min: int = 1
    n_mutations_max: int = 3
    max_steps: int = 1_000_000
    timeout: int = -1
    algo: str = "astar"
    seed: int = 42
    fitness: str = "states"
    allow_player_change: bool = False
    render_gif: bool = True
    gif_frame_duration: float = 0.05
    gif_scale: int = 1
    n_workers: int = 4
    depth_increase_threshold: float = 0.95


@dataclass
class ExitTrainConfig:
    game: Optional[str] = None
    level: Optional[int] = None
    all_games: bool = False
    random_order: bool = False
    iterations: int = 200
    max_nodes: int = 100_000
    batch_size: int = 1000
    cost_weight: float = 0.6
    train_steps_per_iter: int = 200
    train_batch_size: int = 256
    lr: float = 1e-3
    blend_alpha: float = 0.5
    replay_max_size: int = 200_000
    resume: bool = True
    save_dir: Optional[str] = None
    # Architecture
    initial_dim: int = 512
    hidden_dim: int = 256
    res_n: int = 2
    # SLURM
    slurm: bool = False
    slurm_job_name: str = "puzzlejax-exit"
    slurm_mem_gb: int = 30
    slurm_cpus_per_task: int = 1
    slurm_timeout_min: int = 60 * 24
    slurm_gres: str = "gpu:1"
    slurm_array_parallelism: int = 1000
    n_games_per_job: int = 1


@dataclass
class PlotSearch(SearchNodeJSConfig):
    algo: str = "all"  # 'all', 'bfs', 'astar', 'mcts'
    aggregate: bool = True  # (Re-)collect all the solution JSONs to compile a results dict for plotting


@dataclass
class RLConfig(PSConfig):
    max_episode_steps: int = 200
    lr: float = 1.0e-4
    n_envs: int = 500
    # How many steps do I take in all of my batched environments before doing a gradient update
    num_steps: int = 128
    total_timesteps: int = int(5e7)
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

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: float = 0.05

    hidden_dims: Tuple[int] = (128, 128)

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
    wandb_project: str = 'puzzlejax_ppo'

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
class EvalConfig(TrainConfig):
    reevaluate: bool = True
    random_agent: bool = False
    # In how many bins to divide up each metric being evaluated
    n_bins: int = 10
    n_eps: int = 1
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[float] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    # Which eval metric to keep in our generated table if sweeping over eval hyperparams (in which case we want to 
    # save space). Only applied when running `cross_eval.py`
    metrics_to_keep: Tuple[str] = ('mean_ep_reward',)
    # metrics_to_keep: Tuple[str] = ('mean_fps',)


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
class SweepRLConfig(TrainConfig):
    game: Optional[str] = None
    all_games: bool = False
    plot: bool = False
    success_heatmap: bool = False
    slurm: bool = True
    mode: str = 'train'
    render_ims: bool = False
    sweep_name: str = "learning_rate"
    backend: str = "jax"  # "jax" → train_jax.py; "cpp" / "nodejs" → train_pytorch.py
    sweep_axes: dict = field(default_factory=lambda: {
        "seed": (0, 1, 2, 3, 4),
        "lr": (1.0e-4,),
    })


cs = ConfigStore.instance()
cs.store(name="preprocess_config", node=PreprocessConfig)
cs.store(name="plot_rand_profile_config", node=PlotRandProfileConfig)
cs.store(name="plot_standalone_bfs_config", node=PlotSearch)
cs.store(name="jax_validation_config", node=JaxValidationConfig)
cs.store(name="cpp_validation_config", node=CppValidationConfig)
cs.store(name="config", node=RLConfig)
cs.store(name="bfs_config", node=BFSConfig)
cs.store(name="profile_jax_config", node=ProfileJaxRandConfig)
cs.store(name="search_nodejs_config", node=SearchNodeJSConfig)
cs.store(name="search_cpp_config", node=SearchCppConfig)
cs.store(name="profile_rand_nodejs_config", node=ProfileRandNodeJSConfig)
cs.store(name="profile_rand_cpp_config", node=ProfileRandCppConfig)
cs.store(name="evolve_level_config", node=EvolveLevelConfig)
cs.store(name="evolve_level_nodejs_config", node=EvolveLevelNodeJSConfig)
cs.store(name="evolve_level_cpp_config", node=EvolveLevelCppConfig)
cs.store(name="train_config", node=TrainConfig)
cs.store(name="eval_config", node=EvalConfig)
cs.store(name="enjoy_config", node=EnjoyConfig)
cs.store(name="sweep_rl_config", node=SweepRLConfig)
cs.store(name="exit_train_jax_config", node=ExitTrainConfig)
