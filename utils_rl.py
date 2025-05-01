import dataclasses
from functools import partial
import math
import os
import pickle
import shutil
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

import chex
import imageio
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training import orbax_utils
import numpy as np
from jax_utils import stack_leaves
import optax
import orbax.checkpoint as ocp
import wandb
import functools
from flax.training.train_state import TrainState
import hydra
from omegaconf import DictConfig, OmegaConf
from time import perf_counter

from conf.config import Config, MultiAgentConfig
from env import PSEnv, PSObs, PSState, PSParams, init_ps_env
from marl.model import ActorCategorical, ActorMLP, ActorRNN, CriticRNN, MAConvForward2, ScannedRNN
from models import NCA, AutoEncoder, ConvForward, ConvForward2, SeqNCA, ActorCriticPS, Dense
from purejaxrl.wrappers import LogWrapper

N_AGENTS = 1

def get_exp_dir(config: MultiAgentConfig):
    exp_dir = os.path.join(
        "rl_logs", 
        "game",
        (
            f"n-envs-{config.n_envs}_"
            f"{config.model}-{'-'.join([str(hd) for hd in config.hidden_dims])}_"
            f"seed-{config.seed}"
        )
    )
    return exp_dir

def get_env_params_from_config(config: MultiAgentConfig):
    return PSParams(
        level_i=config.level_i,
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


def init_run(config: MultiAgentConfig, ckpt_manager, latest_update_step, rng):
    # Create PCGRL environment
    env_params = get_env_params_from_config(config)
    env = init_ps_env(env_params)

    # Wrap environment with JAXMARL wrapper
    # env = MultiAgentWrapper(env, env_params)

    # Wrap environment with LogWrapper
    # env = MALogWrapper(env)
    env = LogWrapper(env)

    # Configure training
    config._num_actors = N_AGENTS * config.n_envs
    
    config._num_updates = int(
        config.total_timesteps // config.num_steps // config.n_envs
    )
    config._minibatch_size = (
        config._num_actors * config.num_steps // config.NUM_MINIBATCHES
    )
    config.CLIP_EPS = (
        config.CLIP_EPS / N_AGENTS
        if config.scale_clip_eps
        else config.CLIP_EPS
    )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
 
    if config._is_recurrent:
        actor_network = ActorCategorical(env.action_space(env.agents[0]).n,
                                subnet=ActorRNN(env.action_space(env.agents[0]).n, config=config,
                                #  subnet=ActorMLP(env.action_space(env.agents[0]).shape[0], config=config,
                                                ))
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        # ac_init_x = (
        #     jnp.zeros((1, config.n_envs, env.observation_space(env.agents[0]).shape[0])),
        #     jnp.zeros((1, config.n_envs)),
        #     jnp.zeros((1, config.n_envs, env.action_space(env.agents[0]).n)),
        # )
        # ac_init_hstate = ScannedRNN.initialize_carry(config.n_envs, config.hidden_dims[0])
        ac_init_x, ac_init_hstate = env.gen_dummy_obs(config)
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        print(actor_network.subnet.tabulate(rngs=_rng_actor, x=ac_init_x, hidden=ac_init_hstate))

        cr_init_x = (
            jnp.zeros((1, config.n_envs, env.world_state_size,)),  
            jnp.zeros((1, config.n_envs)),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config.n_envs, config.hidden_dims[0])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        _linear_schedule = partial(linear_schedule, config)
        if config.ANNEAL_LR:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=_linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=_linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )
        train_states = (actor_train_state, critic_train_state)
        ac_init_hstate = ScannedRNN.initialize_carry(config._num_actors, config.hidden_dims[0])
        cr_init_hstate = ScannedRNN.initialize_carry(config._num_actors, config.hidden_dims[0])
    else:
        network = init_network(env, env_params, config)
        actor_network = network

        init_x = env._env._env.gen_dummy_obs(env_params)
        # init_x = env.observation_space(env_params).sample(_rng)[None, ]
        avail_actions = env.get_avail_actions(env_state)
        avail_actions = jax.lax.stop_gradient(
            batchify(avail_actions, env.agents, len(env.agents))
        )[np.newaxis]
        network_params = network.init(rng, init_x, avail_actions=avail_actions)
        print(network.subnet.tabulate(_rng, init_x.map_obs, init_x.flat_obs))
    
        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(config.lr, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        train_states = (train_state,)
        ac_init_hstate = None
        cr_init_hstate = None

    rng, _rng = jax.random.split(rng)
    runner_state = RunnerState(
        train_states,
        env_state,
        obsv,
        jnp.zeros((config._num_actors), dtype=bool, ),
        (ac_init_hstate, cr_init_hstate),
        _rng,
    )

    return runner_state, actor_network, env, latest_update_step


def init_network(env: PSEnv, env_params: PSParams, config: Config):
    action_dim = env.action_space.n

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
        )
    elif config.model == "rnn":
        # TODO: Standardize everything to take and return (by default None/unused) hidden states. Enable multi-agent 
        #   script to use non-RNN networks.
        network = ActorCategorical(action_dim,
                             subnet=ActorRNN(env.action_space(env.agents[0]).n, config=config,
                            #  subnet=ActorMLP(env.action_space(env.agents[0]).shape[0], config=config,
                                             ))
        return network
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

 


def restore_run(config: MultiAgentConfig, runner_state: RunnerState, ckpt_manager, latest_update_step: int, load_wandb: bool = True):
    wandb_run_id=None
    if latest_update_step is not None:
        runner_state = ckpt_manager.restore(latest_update_step, args=ocp.args.StandardRestore(runner_state))
        if load_wandb: 
            with open(os.path.join(config._exp_dir, "wandb_run_id.txt"), "r") as f:
                wandb_run_id = f.read()

    return runner_state, wandb_run_id


def make_sim_render_episode(config: MultiAgentConfig, actor_network, env: PSEnv):
    
    # FIXME: Shouldn't hardcode this
    max_episode_len = env.max_steps
    
    # remaining_timesteps = init_state.env_state.remaining_timesteps
    # actor_params = runner_state.train_states[0].params
    # actor_hidden = runner_state.hstates[0]

    def sim_render_episode(actor_params, actor_hidden):
        rng = jax.random.PRNGKey(0)
        
        init_obs, init_state = env.reset(rng)
        
        def step_env(carry, _):
            rng, obs, state, done, actor_hidden = carry
            # print(obs.shape)

            # traj = datatypes.dynamic_index(
            #     state.env_state.sim_trajectory, state.env_state.timestep, axis=-1, keepdims=True
            # )
            avail_actions = env.get_avail_actions(state.env_state)
            if config._is_recurrent:
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, len(env.agents))
                )
                obs = batchify(obs, env.agents, N_AGENTS)
                ac_in = (
                    obs[np.newaxis, :],
                    # obs,
                    done[np.newaxis, :],
                    # done,
                    avail_actions[np.newaxis, :],
                )
                actor_hidden, pi = actor_network.apply(actor_params, actor_hidden, ac_in)            
            else:
                avail_actions = jax.tree.map(lambda x: x[jnp.newaxis], avail_actions)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, len(env.agents))
                )
                obs = jax.tree.map(lambda x: x[jnp.newaxis], obs)
                obs = batchify(obs, env.agents, N_AGENTS)
                # obs = obs.replace(flat_obs=obs.flat_obs[..., jnp.newaxis])
                pi, _ = actor_network.apply(actor_params, obs, avail_actions)
            action = pi.sample(seed=rng)
            env_act = unbatchify(
                action, env.agents, 1, N_AGENTS
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # outputs = [
            #     jit_select_action({}, state, obs, None, rng)
            #     for jit_select_action in jit_select_action_list
            # ]
            # action = agents.merge_actions(outputs)
            obs, next_state, reward, done, info = env.step(state=state, action=env_act, key=rng)
            rng, _ = jax.random.split(rng)
            done = batchify(done, env.agents, N_AGENTS)[:, 0]

            return (rng, obs, next_state, done, actor_hidden), next_state

            
        done = jnp.zeros((len(env.agents),), dtype=bool)

        _, states = jax.lax.scan(step_env, (rng, init_obs, init_state, done, actor_hidden), None, length=max_episode_len)

        # Concatenate the init_state to the states
        states = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y], axis=0), init_state, states)

        frames = jax.vmap(env.render)(states.env_state)

        return frames

    return jax.jit(sim_render_episode)

# states = []
# rng, obs, state, done, actor_hidden = (rng, init_obs, init_state, done, actor_hidden)
# for i in range(remaining_timesteps):
#     carry, state = step_env((rng, obs, state, done, actor_hidden), None)
#     rng, obs, state, done, actor_hidden = carry
#     states.append(state)

    
def render_callback(env: PSEnv, frames, save_dir: str, t: int, max_steps: int):

    imageio.mimsave(os.path.join(save_dir, f"enjoy_{t}.gif"), np.array(frames), fps=20, loop=0)
    wandb.log({"video": wandb.Video(os.path.join(save_dir, f"enjoy_{t}.gif"), fps=20, format="gif")})


def get_ckpt_dir(config: MultiAgentConfig):
    ckpts_dir = os.path.abspath(os.path.join(config._exp_dir, "ckpts"))
    return ckpts_dir

    
def init_config(config: MultiAgentConfig):
    # config._num_eval_actors = config.n_eval_envs * config.n_agents
    config._exp_dir = get_exp_dir(config)
    config._ckpt_dir = get_ckpt_dir(config)
    config._vid_dir = os.path.join(config._exp_dir, "vids")
    config._n_gpus = jax.local_device_count()
    config._is_recurrent = config.model in {'rnn'}

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config

    
def save_checkpoint(config: MultiAgentConfig, ckpt_manager, runner_state, t):
    save_args = orbax_utils.save_args_from_target(runner_state)
    ckpt_manager.save(t.item(), args=ocp.args.StandardSave(runner_state))
    ckpt_manager.wait_until_finished() 