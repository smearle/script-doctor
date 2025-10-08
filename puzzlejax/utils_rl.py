import os
from timeit import default_timer as timer
from typing import Tuple, Dict

import imageio
import jax
import jax.numpy as jnp
from flax import struct
# from flax.training import orbax_utils
import numpy as np
from puzzlejax.utils_jax import stack_leaves
import orbax.checkpoint as ocp
from puzzlejax.preprocess_games import get_tree_from_txt
import wandb
from flax.training.train_state import TrainState
from time import perf_counter

from puzzlejax.conf.config import RLConfig, TrainConfig
from puzzlejax.env import PSEnv, PSObs, PSState, PSParams
from puzzlejax.models import NCA, AutoEncoder, ConvForward, ConvForward2, SeqNCA, ActorCriticPS, Dense

N_AGENTS = 1

def get_exp_dir(config: TrainConfig):
    exp_dir = os.path.join(
        "rl_logs", 
        f"{config.game}",
        f"level-{config.level}",
        (
            f"n-envs-{config.n_envs}_"
            f"{config.model}-{'-'.join([str(hd) for hd in config.hidden_dims])}_"
            f"seed-{config.seed}"
        )
    )
    return exp_dir

def get_env_params_from_config(env: PSEnv, config: RLConfig):
    level = env.get_level(config.level)
    return PSParams(
        level=level
    )

@struct.dataclass
class RunnerState:
    train_states: Tuple[TrainState, TrainState]
    env_state: PSState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    hstates: Tuple[jnp.ndarray, jnp.ndarray]
    rng: jnp.ndarray


def batchify(x: Dict[int, PSObs], agent_list, num_actors):
    # obs_list = [x[a] for a in agent_list]
    # if isinstance(obs_list[0], PSObs):
    #     x = PSObs(
    #         multihot_level=jnp.stack([obs.multihot_level for obs in obs_list]),
    #         flat_obs=jnp.stack([obs.flat_obs for obs in obs_list]),
    #     )
    #     # x = stack_leaves(obs_list)
    #     # Assume we have a batch dimension and an agent dimension
    #     return jax.tree.map(lambda x: x.reshape((num_actors, *x.shape[2:])), x)
    # else:
    #     x = jnp.stack(obs_list)
    # return x.reshape((num_actors, -1))
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    # x = x.reshape((num_actors, num_envs, -1))
    # return {a: x[i] for i, a in enumerate(agent_list)}
    return x


def linear_schedule(config, count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac



def init_network(env: PSEnv, env_params: PSParams, config: RLConfig):
    action_dim = env.action_space.n

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
        )
    elif config.model == "conv":
        network = ConvForward(
            action_dim=action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "conv2":
        network = ConvForward2(
            action_dim=action_dim, activation=config.activation,
            act_shape=config.act_shape,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "seqnca":
        network = SeqNCA(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model in {"nca", "autoencoder"}:
        if config.model == "nca":
            network = NCA(
                representation=config.representation,
                tile_action_dim=env.rep.tile_action_dim,
                activation=config.activation,
            )
        elif config.model == "autoencoder":
            network = AutoEncoder(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
    else:
        raise Exception(f"Unknown model {config.model}")
    network = ActorCriticPS(network)
    return network

    
def render_callback(env: PSEnv, frames, save_dir: str, t: int, max_steps: int):

    imageio.mimsave(os.path.join(save_dir, f"enjoy_{t}.gif"), np.array(frames), fps=20, loop=0)
    wandb.log({"video": wandb.Video(os.path.join(save_dir, f"enjoy_{t}.gif"), fps=20, format="gif")})


def get_ckpt_dir(config: TrainConfig):
    ckpts_dir = os.path.abspath(os.path.join(config._exp_dir, "ckpts"))
    return ckpts_dir

    
def init_config(config: TrainConfig) -> TrainConfig:
    # config._num_eval_actors = config.n_eval_envs * config.n_agents
    config._exp_dir = get_exp_dir(config)
    config._ckpt_dir = get_ckpt_dir(config)
    config._vid_dir = os.path.join(config._exp_dir, "vids")
    config._n_gpus = jax.local_device_count()
    config._is_recurrent = config.model in {'rnn'}

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config

    
def save_checkpoint(config: TrainConfig, ckpt_manager, runner_state, t):
    # save_args = orbax_utils.save_args_from_target(runner_state)
    ckpt_manager.save(t.item(), args=ocp.args.StandardSave(runner_state))
    ckpt_manager.wait_until_finished() 

import puzzlejax.utils as utils

def init_ps_env(config: RLConfig, verbose: bool = False) -> PSEnv:
    #return utils.init_ps_env(config.game, config.level, config.max_episode_steps, vmap=config.vmap)
    return utils.init_ps_env(game=config.game, level_i=config.level, max_episode_steps=config.max_episode_steps)
