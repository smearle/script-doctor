from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from conf.config import TrainConfig
from puzzlescript_nodejs.rl_env import NodeJSPuzzleEnv
from utils_rl import get_exp_dir, init_config, init_network


@dataclass
class TransitionBatch:
    obs: Any
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray


def _policy_action(network, params, obs: Any, rng: jax.Array):
    pi, value = network.apply(params, obs)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return int(action.item()), log_prob, value


def _obs_batch(obs):
    return jax.tree_util.tree_map(lambda x: x[None], obs)


def _stack_obs(obs_list):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *obs_list)


def _render_rollout(env: NodeJSPuzzleEnv, network, params, config: TrainConfig, update_i: int) -> None:
    obs, state = env.reset()
    actions: list[int] = []
    rng = jax.random.PRNGKey(config.seed + update_i + 1_000_000)
    episode_return = 0.0
    for _ in range(env.max_steps):
        rng, sample_rng = jax.random.split(rng)
        action, _, _ = _policy_action(network, params, _obs_batch(obs), sample_rng)
        actions.append(action)
        obs, state, reward, done, _ = env.step(state, action)
        episode_return += reward
        if done:
            break

    gif_path = os.path.join(config._exp_dir, f"update-{update_i}.gif")
    env.render_gif(
        actions=actions,
        gif_path=gif_path,
        frame_duration_s=config.gif_frame_duration,
        scale=10,
    )
    wandb.log({"video": wandb.Video(gif_path, format="gif"), "render_return": episode_return}, step=update_i)


def make_train(config: TrainConfig):
    if config.n_envs != 1:
        raise ValueError("train_nodejs.py currently supports only n_envs=1.")
    if config.model != "dense":
        raise ValueError("train_nodejs.py currently supports only model=dense.")

    config._num_updates = config.total_timesteps // config.num_steps
    config._minibatch_size = config.num_steps // config.NUM_MINIBATCHES
    if config.num_steps % config.NUM_MINIBATCHES != 0:
        raise ValueError("num_steps must be divisible by NUM_MINIBATCHES for train_nodejs.py.")

    env = NodeJSPuzzleEnv(config.game, config.level, config.max_episode_steps)
    network = init_network(env, None, config)

    def linear_schedule(count):
        frac = 1.0 - (count // (config.NUM_MINIBATCHES * config.update_epochs)) / max(config._num_updates, 1)
        return config.lr * frac

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        init_obs = env.gen_dummy_obs(None)
        network_params = network.init(init_rng, init_obs)
        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(learning_rate=linear_schedule if config.ANNEAL_LR else config.lr, eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        obs, env_state = env.reset()
        episode_return = 0.0
        episode_length = 0
        global_step = 0
        train_start_time = timer()

        progress_path = os.path.join(config._exp_dir, "progress.csv")
        with open(progress_path, "w") as f:
            f.write("timestep,ep_return\n")

        for update_i in range(config._num_updates):
            obs_buf = []
            actions_buf = []
            log_probs_buf = []
            rewards_buf = []
            dones_buf = []
            values_buf = []

            for _ in range(config.num_steps):
                global_step += 1
                obs_buf.append(obs)
                rng, sample_rng = jax.random.split(rng)
                action, log_prob, value = _policy_action(network, train_state.params, _obs_batch(obs), sample_rng)
                next_obs, env_state, reward, done, info = env.step(env_state, action)
                actions_buf.append(action)
                log_probs_buf.append(log_prob.squeeze(0))
                rewards_buf.append(reward)
                dones_buf.append(float(done))
                values_buf.append(value.squeeze(0))
                episode_return += reward
                episode_length += 1
                obs = next_obs

                if done:
                    fps = global_step / max(timer() - train_start_time, 1e-6)
                    print(
                        f"global step={global_step:,}; episodic return mean: {episode_return:,.2f} "
                        f"max: {episode_return:,.2f}, min: {episode_return:,.2f}, "
                        f"episode length: {episode_length:,.2f}, FPS: {fps:,.2f}"
                    )
                    with open(progress_path, "a") as f:
                        f.write(f"{global_step},{episode_return}\n")
                    wandb.log(
                        {
                            "ep_return": episode_return,
                            "ep_return_max": episode_return,
                            "ep_return_min": episode_return,
                            "ep_length": episode_length,
                            "fps": fps,
                            "score": info["score"],
                            "won": float(info["won"]),
                        },
                        step=global_step,
                    )
                    obs, env_state = env.reset()
                    episode_return = 0.0
                    episode_length = 0

            batch = TransitionBatch(
                obs=_stack_obs(obs_buf),
                actions=jnp.asarray(actions_buf),
                log_probs=jnp.asarray(log_probs_buf),
                rewards=jnp.asarray(rewards_buf, dtype=jnp.float32),
                dones=jnp.asarray(dones_buf, dtype=jnp.float32),
                values=jnp.asarray(values_buf, dtype=jnp.float32),
            )

            _, next_value = network.apply(train_state.params, _obs_batch(obs))
            next_value = next_value.squeeze(0)
            advantages = []
            gae = 0.0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    next_non_terminal = 1.0 - batch.dones[t]
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - batch.dones[t + 1]
                    next_values = batch.values[t + 1]
                delta = batch.rewards[t] + config.GAMMA * next_values * next_non_terminal - batch.values[t]
                gae = delta + config.GAMMA * config.GAE_LAMBDA * next_non_terminal * gae
                advantages.append(gae)
            advantages = jnp.asarray(list(reversed(advantages)), dtype=jnp.float32)
            returns = advantages + batch.values

            batch_size = config.num_steps
            obs_flat = batch.obs
            actions_flat = batch.actions
            log_probs_flat = batch.log_probs
            values_flat = batch.values

            def loss_fn(params, mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_values):
                pi, value = network.apply(params, mb_obs)
                log_prob = pi.log_prob(mb_actions)
                ratio = jnp.exp(log_prob - mb_old_log_probs)
                norm_adv = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                loss_actor1 = ratio * norm_adv
                loss_actor2 = jnp.clip(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * norm_adv
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                value_pred_clipped = mb_values + (value - mb_values).clip(-config.CLIP_EPS, config.CLIP_EPS)
                value_losses = jnp.square(value - mb_returns)
                value_losses_clipped = jnp.square(value_pred_clipped - mb_returns)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                entropy = pi.entropy().mean()
                total_loss = loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            for _ in range(config.update_epochs):
                rng, perm_rng = jax.random.split(rng)
                permutation = np.asarray(jax.random.permutation(perm_rng, batch_size))
                for start in range(0, batch_size, config._minibatch_size):
                    mb_inds = permutation[start:start + config._minibatch_size]
                    mb_obs = jax.tree_util.tree_map(lambda x: x[mb_inds], obs_flat)
                    (_, _), grads = grad_fn(
                        train_state.params,
                        mb_obs,
                        actions_flat[mb_inds],
                        log_probs_flat[mb_inds],
                        advantages[mb_inds],
                        returns[mb_inds],
                        values_flat[mb_inds],
                    )
                    train_state = train_state.apply_gradients(grads=grads)

            if config.render_freq > 0 and (update_i + 1) % config.render_freq == 0:
                _render_rollout(env, network, train_state.params, config, update_i + 1)

        params_path = os.path.join(config._exp_dir, "params.npz")
        flat_params = {}
        for key, value in jax.tree_util.tree_flatten_with_path(train_state.params)[0]:
            name = "/".join(str(part.key) if hasattr(part, "key") else str(part) for part in key)
            flat_params[name] = np.asarray(value)
        np.savez(params_path, **flat_params)
        return train_state

    return train


@hydra.main(version_base="1.3", config_path="conf", config_name="train_nodejs")
def main(config: TrainConfig):
    logging.getLogger().setLevel(logging.WARNING)
    config = init_config(config)
    config._exp_dir = f"{get_exp_dir(config)}_nodejs"
    print(f"Running experiment to be logged at {config._exp_dir}\n")

    if config.overwrite and os.path.exists(config._exp_dir):
        shutil.rmtree(config._exp_dir)
    os.makedirs(config._exp_dir, exist_ok=True)

    run = wandb.init(
        project=getattr(config, "wandb_project", "puzzlejax_ppo"),
        config=OmegaConf.to_container(config),
        mode=getattr(config, "wandb_mode", "online"),
        dir=config._exp_dir,
        resume=None,
    )
    with open(os.path.join(config._exp_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run.id)

    rng = jax.random.PRNGKey(config.seed)
    train = make_train(config)
    train(rng)


if __name__ == "__main__":
    main()
