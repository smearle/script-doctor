"""Render GIFs of a trained recurrent (GRU) PPO agent from its latest
checkpoint. Produces both a stochastic-sample and a greedy (argmax) rollout
per episode so we can showcase the "best" learned behavior.
"""
from functools import partial
import os

import hydra
import jax
import jax.numpy as jnp
import imageio
import numpy as np

from conf.config import TrainConfig
from purejaxrl.wrappers import LogWrapper
from puzzlejax.models import ScannedRNN
from puzzlejax.utils_rl import (
    get_env_params_from_config, init_config, init_network, init_ps_env,
)
from puzzlejax.train_jax_rnn import init_checkpointer


def _rollout(network, params, env_r, env_params, n_eps, hidden_dim, rng, greedy):
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, n_eps)
    obsv, env_state = jax.vmap(env_r.reset, in_axes=(0, None))(reset_rng, env_params)
    hstate = ScannedRNN.initialize_carry(n_eps, hidden_dim)
    done = jnp.zeros((n_eps,), dtype=bool)

    def _step(carry, _):
        rng, obs, env_state, done, hstate = carry
        rng, _rng = jax.random.split(rng)
        ac_in = (jax.tree_util.tree_map(lambda x: x[None], obs), done[None])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        if greedy:
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        else:
            action = pi.sample(seed=_rng).squeeze(0)
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, n_eps)
        obs, env_state, reward, done, info = jax.vmap(
            env_r.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
        frames = jax.vmap(partial(env_r.render, cv2=False), in_axes=(0,))(env_state)
        return (rng, obs, env_state, done, hstate), (frames, done, info["won"])

    _, (frames, dones, wons) = jax.lax.scan(
        _step, (rng, obsv, env_state, done, hstate), None, env_r.max_steps)
    return frames, dones, wons


@hydra.main(version_base="1.3", config_path='../conf', config_name='train')
def main(config: TrainConfig):
    config.model = 'rnn'
    config = init_config(config)
    hidden_dim = config.hidden_dims[0]

    env_r = init_ps_env(config, verbose=False)
    env_params = get_env_params_from_config(env_r, config)
    env = LogWrapper(env_r)
    network = init_network(env, env_params, config)

    checkpoint_manager, restored_ckpt, _ = init_checkpointer(config)
    if restored_ckpt is None:
        raise RuntimeError(f"No checkpoint found for {config._exp_dir}")
    params = restored_ckpt['runner_state'].train_state.params
    step = restored_ckpt['steps_prev_complete']

    out_dir = os.path.join(config._exp_dir, "enjoy")
    os.makedirs(out_dir, exist_ok=True)
    rng = jax.random.PRNGKey(config.eval_seed if hasattr(config, 'eval_seed') else 0)

    n_eps = config.n_render_eps
    for greedy in (True, False):
        tag = "greedy" if greedy else "sample"
        rng, _rng = jax.random.split(rng)
        frames, dones, wons = _rollout(
            network, params, env_r, env_params, n_eps, hidden_dim, _rng, greedy)
        # frames: (T, n_eps, H, W, C); wons/dones: (T, n_eps)
        frames = np.asarray(frames)
        wons = np.asarray(wons)
        dones = np.asarray(dones)
        for ep in range(n_eps):
            # Truncate at first episode end so the gif shows a single attempt.
            ep_done = dones[:, ep]
            end = int(np.argmax(ep_done)) + 1 if ep_done.any() else frames.shape[0]
            won = bool(wons[:end, ep].any())
            gif = os.path.join(out_dir, f"{config.game}_{tag}_ep{ep}_{'WIN' if won else 'noWin'}.gif")
            imageio.v3.imwrite(gif, frames[:end, ep], duration=config.gif_frame_duration, loop=0)
            print(f"  saved {gif}  (won={won}, len={end})")
        win_rate = float((wons.cumsum(0) > 0)[-1].mean())
        print(f"[{config.game}] {tag} checkpoint-step={step} win_rate={win_rate:.2f}")


if __name__ == "__main__":
    main()
