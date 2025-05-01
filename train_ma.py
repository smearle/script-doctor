"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""
import dataclasses
from functools import partial
import os
import shutil
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
import numpy as np
import orbax.checkpoint as ocp
import wandb
import functools
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf

from marl.model import ActorRNN, CriticRNN, ScannedRNN
from conf.config import MultiAgentConfig
from utils_rl import RunnerState, batchify, init_run, ma_init_config, make_sim_render_episode, render_callback, restore_run, save_checkpoint, unbatchify, N_AGENTS
from utils import get_env_params_from_config, init_network

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    # control_mask: jnp.ndarray 

def make_train(
    config: MultiAgentConfig, 
    checkpoint_manager,
    actor_network, 
    env,  
    latest_update_step, 
):

    # env_params = get_env_params_from_config(config)
    # render_env = PCGRLEnv(env_params)
    # render_env.init_graphics()
    env.init_graphics()

    # Define which parts of the callback we don't want to trace
    _render_callback = partial(render_callback, save_dir=config._vid_dir, max_steps=env.max_steps, env=env)

    def train(rng, runner_state=None):

        # INIT ENV
        rng, _rng = jax.random.split(rng)

        
        jit_sim_render_episode = make_sim_render_episode(config, actor_network, env)
        num_render_actors = 1 * env.n_agents
        if config._is_recurrent:
            # INIT NETWORK
            critic_network = CriticRNN(config=config)
            ac_init_hstate_render = ScannedRNN.initialize_carry(num_render_actors, config.hidden_dims[0])
        else:
            network = actor_network
            ac_init_hstate_render = None

        render_frames = jit_sim_render_episode(runner_state.train_states[0].params, ac_init_hstate_render)
                
        # DEFINE CALLBACKS
        jax.experimental.io_callback(callback=_render_callback, result_shape_dtypes=None, frames=render_frames, t=0)
        
        # TRAIN LOOP
        def _update_step_with_render(update_runner_state, unused, render_frames):
            
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state: RunnerState, unused):
                
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state.train_states,
                    runner_state.env_state,
                    runner_state.last_obs,
                    runner_state.last_done,
                    runner_state.hstates, 
                    runner_state.rng
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config._num_actors)
                )
                # control_mask = jax.vmap(env._get_control_mask)(env_state.env_state)
                
                obs_batch = batchify(last_obs, env.agents, config._num_actors)
    
                if config._is_recurrent:
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                        avail_actions,
                    )
                    ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)

                    # VALUE
                    # output of wrapper is (num_envs, num_agents, world_state_size)
                    # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                    world_state = last_obs["world_state"].swapaxes(0, 1)  
                    # shape: (num_envs * num_max_objects, world_state_size)
                    world_state = world_state.reshape((config._num_actors, -1))
                
                    cr_in = (
                        world_state[None, :],
                        last_done[np.newaxis, :],
                    )
                    cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                    hstates = (ac_hstate, cr_hstate)
                else:
                    # obs_batch = jax.tree.map(lambda x: x[np.newaxis], obs_batch)
                    # obs_batch_in = obs_batch.replace(flat_obs = obs_batch.flat_obs[..., np.newaxis])
                    pi, value = network.apply(train_states[0].params, obs_batch, avail_actions)
                    world_state = jax.tree.map(lambda x: x.swapaxes(0, 1), last_obs["world_state"])
                    # shape: (num_envs * num_max_objects, world_state_size)
                    world_state = jax.tree.map(lambda x: x.reshape((config._num_actors, -1)), world_state)
                    hstates = (None, None)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config.n_envs, env.n_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.n_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config._num_actors)), info)
                done_batch = batchify(done, env.agents, config._num_actors).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.n_agents),
                    last_done,
                    # action.squeeze(),
                    action,
                    # value.squeeze(),
                    value,
                    batchify(reward, env.agents, config._num_actors).squeeze(),
                    # log_prob.squeeze(),
                    log_prob,
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                    # control_mask,
                )
                runner_state = RunnerState(train_states, env_state, obsv, done_batch, hstates, rng)
                return runner_state, transition

            initial_hstates = runner_state.hstates
            
            # DO ROLLOUTS
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )
      
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = (
                runner_state.train_states,
                runner_state.env_state,
                runner_state.last_obs,
                runner_state.last_done,
                runner_state.hstates, 
                runner_state.rng, 
            )

            if config._is_recurrent:
                last_world_state = last_obs["world_state"].swapaxes(0,1)
                last_world_state = last_world_state.reshape((config._num_actors,-1))
            
                cr_in = (
                    last_world_state[None, :],
                    last_done[np.newaxis, :],
                )
                _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            else:
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config._num_actors)
                )
                obs_batch = batchify(last_obs, env.agents, config._num_actors)
                _, last_val = network.apply(train_states[0].params, obs_batch, avail_actions)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                                
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):

                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    train_state = train_state[0]

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # obs = traj_batch.obs[None]
                        pi, value = network.apply(params, traj_batch.obs, traj_batch.avail_actions)
                        # action = traj_batch.action.reshape(pi.logits.shape[:-1])
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses,
                                              value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        # Some reshaping to accomodate player, x, and y dimensions to action output. (Not used often...)
                        gae = gae[..., None, None]

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.VF_COEF * value_loss
                            - config.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "value_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                    }
                    return (train_state,), loss_info

                def _update_minibatch_ma(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info
                    
                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
              
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        
                        # Average
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)
                    
                    # COMPUTE ACTOR AND CRITIC GRADIENTS
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
                    # INSERT GRADIENTS
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(
                    x, (1, config._num_actors, -1)
                ), init_hstates)
                
                if not config._is_recurrent:
                    # batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                    batch_size = config.num_steps * config.n_envs * config.n_agents
                    assert (
                        batch_size == config.num_steps * config.n_envs * config.n_agents
                    ), "batch size must be equal to number of steps * number " + \
                        "of envs * number of agents"
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (traj_batch, advantages, targets)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                    )
                else:
                    batch = (
                        init_hstates[0],
                        init_hstates[1],
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )
                    permutation = jax.random.permutation(_rng, config._num_actors)

                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )
                    
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                _update_minibatch_fn = _update_minibatch_ma if config._is_recurrent else _update_minibatch
                
                train_states, loss_info = jax.lax.scan(
                    _update_minibatch_fn, train_states, minibatches
                )
                      
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info
            
            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            if config._is_recurrent:
                loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            
            train_states = update_state[0]
            metric = traj_batch.info
          
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config.num_steps, config.n_envs, env.n_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info

            if config._is_recurrent:
                obs = traj_batch.obs
            else:
                obs = jnp.concatenate((traj_batch.obs.map_obs.flatten(), traj_batch.obs.flat_obs.flatten()), axis=-1)
     
            metric["obs_dist_min"] = obs.min()
            metric["obs_dist_max"] = obs.max()
            metric["obs_dist_std"] = obs.std()
            metric["act_dist_mean"] = traj_batch.action.mean()
            metric["act_dist_std"] = traj_batch.action.std()
            
            rng = update_state[-1]
            
            def log_callback(metric):
                wandb.log(
                    {
                        # the metrics have an agent dimension, but this is identical
                        # for all agents so index into the 0th item of that dimension.
                        "returns": metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0]
                        ].mean(),
                        "env_step": metric["update_steps"]
                        * config._num_actors,
                        **metric["loss"],
                        "obs_dist_min": metric["obs_dist_min"],
                        "obs_dist_max": metric["obs_dist_max"],
                        "obs_dist_std": metric["obs_dist_std"],
                        "act_dist_mean": metric["act_dist_mean"],
                        "act_dist_std": metric["act_dist_std"],
                    }
                )

                print(f"Step: {metric['update_steps']}, Returns: {metric['returned_episode_returns'][:, :, 0][metric['returned_episode'][:, :, 0]].mean()}")

            def ckpt_callback(metric, runner_state):
                try:
                    curr_update_step = metric["update_steps"]
                    save_checkpoint(config, checkpoint_manager, runner_state, curr_update_step)
                except jax.errors.ConcretizationTypeError:
                    return
            
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(log_callback, None, metric)
            runner_state = RunnerState(train_states, env_state, last_obs, last_done, hstates, rng)
            do_render = (config.render_freq != -1) and (update_steps % config.render_freq == 0)
            
            frames = jax.lax.cond(
                do_render,
                partial(jit_sim_render_episode, runner_state.train_states[0].params, ac_init_hstate_render),
                lambda: render_frames,
            )
            jax.lax.cond(
                do_render,
                partial(jax.experimental.io_callback, _render_callback, None, frames=frames, t=update_steps),
                lambda: None,
            )
            jax.lax.cond(
                update_steps % config.ckpt_freq == 0,
                partial(jax.experimental.io_callback, ckpt_callback, None, metric, runner_state),
                lambda: None,
            )
                        
            update_steps = update_steps + 1
            return (runner_state, update_steps), metric
        
        
        _update_step = functools.partial(_update_step_with_render, render_frames=render_frames)
        
        runner_state, metric = jax.lax.scan(
            _update_step,   
            (runner_state, latest_update_step), None, config._num_updates - latest_update_step
        )
        
        return {"runner_state": runner_state} 

    return train
    
@hydra.main(version_base="1.3", config_path="conf", config_name="ma_config")
def main(config: MultiAgentConfig):
    ma_init_config(config)
    
    if config.overwrite:
        shutil.rmtree(config._exp_dir, ignore_errors=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        config._ckpt_dir, 
        # ocp.PyTreeCheckpointer(), 
        options=options)

    rng = jax.random.PRNGKey(config.seed)
    latest_update_step = checkpoint_manager.latest_step()
    
    runner_state, actor_network, env, latest_update_step = \
        init_run(config, checkpoint_manager, latest_update_step, rng)
    
    if latest_update_step is not None:
        runner_state, wandb_run_id = restore_run(config, runner_state, checkpoint_manager, latest_update_step)
        wandb_resume = "Must"
    else:
        wandb_run_id, wandb_resume = None, None

    latest_update_step = 0 if latest_update_step is None else latest_update_step

    os.makedirs(config._exp_dir, exist_ok=True)
    os.makedirs(config._vid_dir, exist_ok=True)

    run = wandb.init(
        project=config.wandb_project,
        tags=["MAPPO"],
        config=OmegaConf.to_container(config),
        mode=config.wandb_mode,
        dir=config._exp_dir,
        id=wandb_run_id,
        resume=wandb_resume,
    )
    wandb_run_id = run.id
    with open(os.path.join(config._exp_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run_id)
    
    with jax.disable_jit(False):
        
        train_jit = jax.jit(
            make_train(
                config=config, 
                checkpoint_manager=checkpoint_manager, 
                env=env, 
                actor_network=actor_network,
                latest_update_step=latest_update_step, 
            )
        ) 
        
        out = train_jit(rng, runner_state=runner_state)

    runner_state = out["runner_state"]
    n_updates = runner_state[-1]
    runner_state: RunnerState = runner_state[0]

if __name__=="__main__":
    main()