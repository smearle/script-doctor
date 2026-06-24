"""Recurrent (GRU) PPO for PuzzleScript, adapted from PureJaxRL's ppo_rnn and
the feed-forward puzzlejax/train_jax.py. The hidden state is threaded through
the rollout and the PPO update, with per-step carry resets at episode
boundaries so the agent's memory is reset on each new attempt. Minibatching is
over the environment axis (time order preserved) so the GRU can be re-scanned.
"""
from functools import partial
import logging
import os
import shutil
from timeit import default_timer as timer
from typing import Any, NamedTuple, Tuple

import hydra
import jax
import jax.numpy as jnp
from flax import struct
import imageio
from omegaconf import OmegaConf
import optax
from flax.training.train_state import TrainState
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm
import wandb

from conf.config import TrainConfig
from purejaxrl.wrappers import LogWrapper, RestartActionWrapper
from puzzlejax.models import ScannedRNN
from puzzlejax.train_jax import log_callback, _render_frames, _render_frames_io
from puzzlejax.utils_rl import (
    get_ckpt_dir, get_env_params_from_config, init_config, init_network,
    init_ps_env,
)


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: Any
    last_obs: Any
    last_done: jnp.ndarray
    hstate: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    # NOTE: `done` here is the *previous* step's done flag (aligned with `obs`),
    # so it doubles as the RNN reset signal for the current observation.
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: Any


def step_env_render_rnn(carry, _, network, n_render_eps, env_r):
    rng, obs, env_state, done, hstate, network_params, env_params = carry
    rng, _rng = jax.random.split(rng)

    ac_in = (jax.tree_util.tree_map(lambda x: x[None], obs), done[None])
    hstate, pi, value = network.apply(network_params, hstate, ac_in)
    action = pi.sample(seed=_rng).squeeze(0)

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, n_render_eps)
    vmap_step_fn = jax.vmap(env_r.step, in_axes=(0, 0, 0, None))
    obs, env_state, reward, done, info = vmap_step_fn(rng_step, env_state, action, env_params)

    vmap_render_fn = jax.vmap(partial(env_r.render, cv2=False), in_axes=(0,))
    frames = vmap_render_fn(env_state)
    return (rng, obs, env_state, done, hstate, network_params, env_params), \
        (env_state, reward, done, info, frames)


def _render_episodes_rnn(network_params, rng_r, network, env_r, n_render_eps, env_params, hidden_dim):
    rng_r, _rng_r = jax.random.split(rng_r)
    reset_rng_r = jax.random.split(_rng_r, n_render_eps)
    vmap_reset_fn = jax.vmap(env_r.reset, in_axes=(0, None))
    obsv_r, env_state_r = vmap_reset_fn(reset_rng_r, env_params)

    init_hstate = ScannedRNN.initialize_carry(n_render_eps, hidden_dim)
    init_done = jnp.zeros((n_render_eps,), dtype=bool)

    _step = partial(step_env_render_rnn, network=network, env_r=env_r, n_render_eps=n_render_eps)
    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        _step,
        (rng_r, obsv_r, env_state_r, init_done, init_hstate, network_params, env_params),
        None, 1 * env_r.max_steps)

    frames = jnp.concatenate(jnp.stack(frames, axis=1))
    return frames, states


def make_train(config: TrainConfig, restored_ckpt, checkpoint_manager):
    config._num_updates = config.total_timesteps // config.num_steps // config.n_envs
    config._minibatch_size = config.n_envs * config.num_steps // config.NUM_MINIBATCHES
    hidden_dim = config.hidden_dims[0]

    env_r = init_ps_env(config, verbose=False)
    env_params = get_env_params_from_config(env_r, config)
    base_env = RestartActionWrapper(env_r) if config.restart_action else env_r
    env = LogWrapper(base_env)
    obs_shape = env_r.observation_space(env_params).shape

    def linear_schedule(count):
        frac = 1.0 - (count // (config.NUM_MINIBATCHES * config.update_epochs)) / config._num_updates
        return config.lr * frac

    def train(rng, config: TrainConfig):
        train_start_time = timer()

        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        from puzzlescript_jax.env import PSObs
        init_obs = PSObs(multihot_level=jnp.zeros((1, 1) + obs_shape), flat_obs=None)
        init_done = jnp.zeros((1, 1), dtype=bool)
        init_carry = ScannedRNN.initialize_carry(1, hidden_dim)
        network_params = network.init(_rng, init_carry, (init_obs, init_done))

        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config.n_envs, hidden_dim)
        last_done = jnp.zeros((config.n_envs,), dtype=bool)

        render_episodes = partial(
            _render_episodes_rnn, env_r=env_r, network=network,
            n_render_eps=config.n_render_eps, env_params=env_params, hidden_dim=hidden_dim)
        render_frames_io = partial(_render_frames_io, env=env_r, config=config)

        rng, _rng = jax.random.split(rng)
        steps_prev_complete = 0
        runner_state = RunnerState(
            train_state, env_state, obsv, last_done, init_hstate, rng, update_i=0)

        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = config.total_timesteps - steps_prev_complete
            config._num_updates = int(steps_remaining // config.num_steps // config.n_envs)

        pbar = tqdm(total=config._num_updates, desc="Training (rnn)", unit="update",
                    dynamic_ncols=True, position=0)
        stats_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
        _log_callback = partial(log_callback, config=config,
                                train_start_time=train_start_time,
                                steps_prev_complete=steps_prev_complete,
                                pbar=pbar, stats_bar=stats_bar,
                                n_levels=len(env_r.levels),
                                csv_path=os.path.join(config._exp_dir, "progress.csv"))

        def save_checkpoint(runner_state, info, steps_prev_complete):
            timesteps = info["timestep"][info["returned_episode"]] * config.n_envs
            if len(timesteps) > 0:
                t = timesteps[-1].item()
                latest_ckpt_step = checkpoint_manager.latest_step()
                if latest_ckpt_step is None or t - latest_ckpt_step >= config.ckpt_freq:
                    print(f"Saving checkpoint at step {t}")
                    ckpt = {'runner_state': runner_state, 'step_i': jnp.array(t, dtype=jnp.int32)}
                    checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))

        def _update_step(runner_state, unused):
            def _env_step(runner_state: RunnerState, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_i = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.last_obs, runner_state.last_done,
                    runner_state.hstate, runner_state.rng, runner_state.update_i,
                )
                rng, _rng = jax.random.split(rng)

                ac_in = (jax.tree_util.tree_map(lambda x: x[None], last_obs), last_done[None])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(action[None]).squeeze(0)
                value = value.squeeze(0)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.n_envs)
                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
                obsv, env_state, reward, done, info = vmap_step_fn(
                    rng_step, env_state, action, env_params)

                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = RunnerState(
                    train_state, env_state, obsv, done, hstate, rng, update_i=update_i)
                return runner_state, transition

            init_hstate = runner_state.hstate  # carry at the start of the rollout
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # CALCULATE ADVANTAGE
            train_state = runner_state.train_state
            ac_in = (jax.tree_util.tree_map(lambda x: x[None], runner_state.last_obs),
                     runner_state.last_done[None])
            _, _, last_val = network.apply(train_state.params, runner_state.hstate, ac_in)
            last_val = last_val.squeeze(0)
            last_done = runner_state.last_done

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.GAMMA * next_value * (1 - next_done) - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params, init_hstate.squeeze(0),
                            (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (loss_actor + config.VF_COEF * value_loss
                                      - config.ENT_COEF * entropy)
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.n_envs)
                # init_hstate: (n_envs, H) -> (1, n_envs, H) so env axis is axis 1
                batch = (init_hstate[None, :], traj_batch, advantages, targets)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config.NUM_MINIBATCHES, -1] + list(x.shape[2:])),
                        1, 0),
                    shuffled_batch)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            rng = runner_state.rng
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs)
            train_state = update_state[0]
            rng = update_state[-1]
            metric = traj_batch.info

            jax.debug.callback(save_checkpoint, runner_state, metric, steps_prev_complete)

            should_render = ((config.render_freq > 0) &
                             (runner_state.update_i % config.render_freq == 0))

            def _do_render():
                render_rng = jax.random.fold_in(runner_state.rng, runner_state.update_i)
                new_frames, _ = render_episodes(train_state.params, render_rng)
                return jax.experimental.io_callback(
                    callback=render_frames_io,
                    result_shape_dtypes=jax.ShapeDtypeStruct((), jnp.int32),
                    frames=new_frames, i=runner_state.update_i, metric=metric,
                    steps_prev_complete=steps_prev_complete)

            render_token = jax.lax.cond(should_render, _do_render, lambda: jnp.int32(0))
            jax.debug.callback(_log_callback, metric)

            runner_state = RunnerState(
                train_state, runner_state.env_state, runner_state.last_obs,
                runner_state.last_done, runner_state.hstate, rng,
                update_i=runner_state.update_i + 1 + (render_token * 0))
            return runner_state, None

        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, config._num_updates)
        return {"runner_state": runner_state}

    return lambda rng: train(rng, config)


def init_checkpointer(config: TrainConfig) -> Tuple[Any, dict, Any]:
    rng = jax.random.PRNGKey(30)
    ckpt_dir = get_ckpt_dir(config)
    hidden_dim = config.hidden_dims[0]

    env_r = init_ps_env(config)
    env_params = get_env_params_from_config(env_r, config)
    base_env = RestartActionWrapper(env_r) if config.restart_action else env_r
    env = LogWrapper(base_env)
    obs_shape = env_r.observation_space(env_params).shape

    rng, _rng = jax.random.split(rng)
    network = init_network(env, env_params, config)
    from puzzlescript_jax.env import PSObs
    init_obs = PSObs(multihot_level=jnp.zeros((1, 1) + obs_shape), flat_obs=None)
    init_done = jnp.zeros((1, 1), dtype=bool)
    network_params = network.init(_rng, ScannedRNN.initialize_carry(1, hidden_dim),
                                  (init_obs, init_done))
    tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                     optax.adam(config.lr, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    init_hstate = ScannedRNN.initialize_carry(config.n_envs, hidden_dim)
    last_done = jnp.zeros((config.n_envs,), dtype=bool)
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv,
                               last_done=last_done, hstate=init_hstate, rng=rng, update_i=0)
    target = {'runner_state': runner_state, 'step_i': jnp.array(0, dtype=jnp.int32)}
    ckpt_dir = os.path.abspath(ckpt_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(ckpt_dir, options=options)

    def try_load_ckpt(steps_prev_complete, target):
        restored_ckpt = checkpoint_manager.restore(
            steps_prev_complete, args=ocp.args.StandardRestore(target))
        restored_ckpt['steps_prev_complete'] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")
        return restored_ckpt

    wandb_run_id = None
    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        ckpt_subdirs = os.listdir(ckpt_dir)
        ckpt_steps = sorted([int(cs) for cs in ckpt_subdirs if cs.isdigit()], reverse=True)
        restored_ckpt = None
        for steps_prev_complete in ckpt_steps:
            try:
                restored_ckpt = try_load_ckpt(steps_prev_complete, target)
                with open(os.path.join(config._exp_dir, "wandb_run_id.txt"), "r") as f:
                    wandb_run_id = f.read()
                break
            except TypeError as e:
                print(f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}")
                continue

    return checkpoint_manager, restored_ckpt, wandb_run_id


@hydra.main(version_base="1.3", config_path='../conf', config_name='train')
def main(config: TrainConfig):
    logging.getLogger().setLevel(logging.WARNING)
    config.model = 'rnn'
    config = init_config(config)
    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config._exp_dir
    print(f'Running recurrent experiment to be logged at {exp_dir}\n')

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    checkpoint_manager, restored_ckpt, wandb_run_id = init_checkpointer(config)
    os.makedirs(exp_dir, exist_ok=True)

    run = wandb.init(
        project=getattr(config, "wandb_project", "puzzlejax_ppo"),
        config=OmegaConf.to_container(config),
        mode=getattr(config, "wandb_mode", "online"),
        dir=exp_dir, id=wandb_run_id, resume=None)
    wandb_run_id = run.id
    with open(os.path.join(exp_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run_id)

    if restored_ckpt is None:
        _env_info = init_ps_env(config, verbose=False)
        n_levels = len(_env_info.levels)
        win_cols = ",".join(f"level-{i}-win" for i in range(n_levels))
        sol_cols = ",".join(f"level-{i}-min_sol_len" for i in range(n_levels))
        with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
            f.write(f"timestep,ep_return,ep_return_max,ep_length,fps,{win_cols},{sol_cols}\n")

    train_jit = jax.jit(make_train(config, restored_ckpt, checkpoint_manager))
    out = train_jit(rng)
    jax.block_until_ready(out)


if __name__ == "__main__":
    main()
