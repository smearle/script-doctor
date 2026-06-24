"""Control-permutation meta-RL (RL^2) on a single PuzzleScript game.

One game is run under N different action->direction permutations. The board
looks identical across variants, but the controls differ, so the agent cannot
tell which variant it is in from observation alone -- it must PROBE (take an
action, observe the effect). A meta-episode is `trials_per_meta` trials: the
level resets each trial but the variant is FIXED and the RNN carry PERSISTS
across trials, so the agent can identify its controls in trial 0 and exploit
them in later trials. Few-shot success => win rate rises with trial index.

Single batched env (one env.step graph) => fast compile, unlike the N-game
generalist. The agent gets NO variant label (pure obs + memory).

Usage:
    python -m puzzlejax.train_jax_rnn_meta game=sokoban_basic level=-1 \
        trials_per_meta=4 n_control_variants=4 max_episode_steps=64 \
        total_timesteps=6000000 wandb_mode=disabled
Set n_control_variants=1 for the no-ambiguity control.
"""
from functools import partial
import logging
import os
import shutil
from timeit import default_timer as timer
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
from flax import struct
from omegaconf import OmegaConf
import optax
from flax.training.train_state import TrainState
import numpy as np
from tqdm import tqdm
import wandb

from conf.config import TrainConfig
from purejaxrl.wrappers import LogWrapper
from puzzlejax.models import ScannedRNN, ActorCriticRNN
from puzzlejax.utils_rl import get_env_params_from_config, init_config, init_ps_env
from puzzlescript_jax.env import PSObs


# A fixed set of direction permutations over actions [0,1,2,3] (the 4 moves);
# action 4 (wait/act) is left unchanged. Variant 0 is identity.
_PERMS = np.array([
    [0, 1, 2, 3, 4],   # identity
    [1, 0, 3, 2, 4],   # swap up<->down, left<->right (180-ish)
    [2, 3, 1, 0, 4],   # rotate
    [3, 2, 0, 1, 4],   # another rotation/mirror
    [0, 2, 1, 3, 4],   # swap two
    [3, 0, 1, 2, 4],
    [1, 2, 3, 0, 4],
    [2, 0, 3, 1, 4],
], dtype=np.int32)


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: Any
    last_obs: Any
    last_done: jnp.ndarray
    trial_count: jnp.ndarray
    hstate: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: Any


def meta_log_callback(metric, csv_path, pbar, stats_bar, n_envs, train_start_time,
                      steps_prev_complete, K):
    done_mask = np.asarray(metric["returned_episode"])
    pbar.update(1)
    if done_mask.sum() == 0:
        return
    t = int(np.asarray(metric["timestep"])[done_mask][-1] * n_envs)
    won = np.asarray(metric["won"])[done_mask].astype(bool)
    agg = float(won.mean())
    payload = {"win_rate/_mean": agg}
    fewshot_str = ""
    if K > 1 and "trial_idx" in metric:
        tidx = np.asarray(metric["trial_idx"])[done_mask]
        fs = []
        for k in range(K):
            mk = tidx == k
            v = float(won[mk].mean()) if mk.sum() > 0 else float("nan")
            fs.append(v)
            payload[f"fewshot/trial_{k}"] = v
        fewshot_str = " | fewshot " + " ".join(f"t{k}={fs[k]:.2f}" for k in range(K))
    wandb.log(payload, step=t)
    fps = (t - steps_prev_complete) / max(timer() - train_start_time, 1e-6)
    stats_bar.set_description_str(f"  t={t:,} winrate={agg:.2f} FPS={fps:,.0f}{fewshot_str}")
    with open(csv_path, "a") as f:
        fs_cols = ("," + ",".join(f"{v:.4f}" for v in fs)) if K > 1 else ""
        f.write(f"{t},{agg}{fs_cols}\n")


def make_train(config: TrainConfig):
    N = int(config.n_control_variants)
    K = int(config.trials_per_meta)
    hidden_dim = config.hidden_dims[0]
    env_r = init_ps_env(config, verbose=False)
    env_params = get_env_params_from_config(env_r, config)
    env = LogWrapper(env_r)
    obs_shape = env_r.observation_space(env_params).shape
    n_envs = config.n_envs
    config._num_updates = config.total_timesteps // config.num_steps // n_envs
    config._minibatch_size = n_envs * config.num_steps // config.NUM_MINIBATCHES

    perm_table = jnp.asarray(_PERMS[:N])             # (N, 5)
    variant_ids = jnp.arange(n_envs) % N             # per-env control variant
    print(f"control-perm meta-RL: game={config.game} N_variants={N} K={K} "
          f"per-trial-budget={config.max_episode_steps} obs={obs_shape}")

    def train(rng, config):
        train_start_time = timer()
        network = ActorCriticRNN(action_dim=5, hidden_dim=hidden_dim, activation=config.activation)
        rng, _rng = jax.random.split(rng)
        init_obs = PSObs(multihot_level=jnp.zeros((1, 1) + obs_shape), flat_obs=None)
        params = network.init(_rng, ScannedRNN.initialize_carry(1, hidden_dim),
                              (init_obs, jnp.zeros((1, 1), dtype=bool)))
        tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                         optax.adam(config.lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(_rng, n_envs), env_params)
        hstate = ScannedRNN.initialize_carry(n_envs, hidden_dim)
        runner_state = RunnerState(train_state, env_state, obsv,
                                   jnp.zeros((n_envs,), dtype=bool),
                                   jnp.zeros((n_envs,), dtype=jnp.int32), hstate, rng, 0)

        pbar = tqdm(total=config._num_updates, desc="meta-rl", position=0, dynamic_ncols=True)
        stats_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
        _log_cb = partial(meta_log_callback, csv_path=os.path.join(config._exp_dir, "progress.csv"),
                          pbar=pbar, stats_bar=stats_bar, n_envs=n_envs,
                          train_start_time=train_start_time, steps_prev_complete=0, K=K)

        def _update_step(runner_state, unused):
            def _env_step(rs, unused):
                rng, _rng = jax.random.split(rs.rng)
                ac_in = (jax.tree_util.tree_map(lambda x: x[None], rs.last_obs), rs.last_done[None])
                hstate, pi, value = network.apply(rs.train_state.params, rs.hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(action[None]).squeeze(0)
                value = value.squeeze(0)
                # Map the agent's action through this env's control permutation.
                env_action = perm_table[variant_ids, action]
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, n_envs), rs.env_state, env_action, env_params)
                new_tc = rs.trial_count + done.astype(jnp.int32)
                meta_done = jnp.logical_and(done, (new_tc % K) == 0)
                tc_next = jnp.where(meta_done, jnp.int32(0), new_tc)
                info = {**info, "trial_idx": rs.trial_count}
                transition = Transition(rs.last_done, action, value, reward, log_prob, rs.last_obs, info)
                rs = RunnerState(rs.train_state, env_state, obsv, meta_done, tc_next, hstate, rng, rs.update_i)
                return rs, transition

            init_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            ac_in = (jax.tree_util.tree_map(lambda x: x[None], runner_state.last_obs),
                     runner_state.last_done[None])
            _, _, last_val = network.apply(runner_state.train_state.params, runner_state.hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _gae(traj, last_val, last_done):
                def f(carry, tr):
                    gae, nv, nd = carry
                    delta = tr.reward + config.GAMMA * nv * (1 - nd) - tr.value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - nd) * gae
                    return (gae, tr.value, tr.done), gae
                _, adv = jax.lax.scan(f, (jnp.zeros_like(last_val), last_val, last_done),
                                      traj, reverse=True, unroll=16)
                return adv, adv + traj.value
            advantages, targets = _gae(traj_batch, last_val, runner_state.last_done)

            def _update_epoch(update_state, unused):
                def _mb(train_state, batch):
                    init_hstate, traj, adv, targ = batch
                    def _loss(p, init_hstate, traj, gae, targ):
                        _, pi, value = network.apply(p, init_hstate.squeeze(0), (traj.obs, traj.done))
                        logp = pi.log_prob(traj.action)
                        vclip = traj.value + (value - traj.value).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        vloss = 0.5 * jnp.maximum(jnp.square(value - targ), jnp.square(vclip - targ)).mean()
                        ratio = jnp.exp(logp - traj.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        la = -jnp.minimum(ratio * gae,
                                          jnp.clip(ratio, 1 - config.CLIP_EPS, 1 + config.CLIP_EPS) * gae).mean()
                        ent = pi.entropy().mean()
                        return la + config.VF_COEF * vloss - config.ENT_COEF * ent, None
                    (loss, _), grads = jax.value_and_grad(_loss, has_aux=True)(
                        train_state.params, init_hstate, traj, adv, targ)
                    return train_state.apply_gradients(grads=grads), loss
                train_state, init_hstate, traj, adv, targ, rng = update_state
                rng, _rng = jax.random.split(rng)
                perm = jax.random.permutation(_rng, n_envs)
                batch = (init_hstate[None, :], traj, adv, targ)
                shuf = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), batch)
                mbs = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(jnp.reshape(
                        x, [x.shape[0], config.NUM_MINIBATCHES, -1] + list(x.shape[2:])), 1, 0), shuf)
                train_state, loss = jax.lax.scan(_mb, train_state, mbs)
                return (train_state, init_hstate, traj, adv, targ, rng), loss

            rng = runner_state.rng
            update_state = (runner_state.train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            train_state = update_state[0]
            rng = update_state[-1]
            jax.debug.callback(_log_cb, traj_batch.info)
            runner_state = RunnerState(train_state, runner_state.env_state, runner_state.last_obs,
                                       runner_state.last_done, runner_state.trial_count,
                                       runner_state.hstate, rng, runner_state.update_i + 1)
            return runner_state, None

        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, config._num_updates)
        return {"runner_state": runner_state}

    return lambda rng: train(rng, config)


@hydra.main(version_base="1.3", config_path='../conf', config_name='train')
def main(config: TrainConfig):
    logging.getLogger().setLevel(logging.WARNING)
    config.model = 'rnn'
    config = init_config(config)
    # distinct exp dir
    config._exp_dir = config._exp_dir + f"_permmeta_N{config.n_control_variants}_K{config.trials_per_meta}"
    rng = jax.random.PRNGKey(config.seed)
    print(f"logging at {config._exp_dir}")
    if config.overwrite and os.path.exists(config._exp_dir):
        shutil.rmtree(config._exp_dir)
    os.makedirs(config._exp_dir, exist_ok=True)
    wandb.init(project=getattr(config, "wandb_project", "puzzlejax_ppo"),
               config=OmegaConf.to_container(config),
               mode=getattr(config, "wandb_mode", "online"), dir=config._exp_dir)
    K = int(config.trials_per_meta)
    with open(os.path.join(config._exp_dir, "progress.csv"), "w") as f:
        cols = "".join(f",fewshot_t{k}" for k in range(K)) if K > 1 else ""
        f.write(f"timestep,win_rate_mean{cols}\n")
    out = jax.jit(make_train(config))(rng)
    jax.block_until_ready(out)


if __name__ == "__main__":
    main()
