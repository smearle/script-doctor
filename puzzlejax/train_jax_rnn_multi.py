"""Generalist recurrent (GRU) PPO across multiple PuzzleScript games.

A single shared network is trained on several games at once. Parallel envs are
partitioned into equal per-game blocks (a tuple of heterogeneous env states);
each game's multi-hot observation is padded to a common (C,H,W); the shared GRU
net is conditioned on a learned per-game embedding (a lookup-table stand-in for
a richer rule encoding). Game id is carried in obs.flat_obs so it travels with
each trajectory through the PPO minibatching.

Usage:
    python -m puzzlejax.train_jax_rnn_multi game=win4 n_envs=256 \
        total_timesteps=6000000 wandb_mode=disabled
where `game` names a preset in MULTI_GAME_PRESETS (or a '+'-joined game list).
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
from omegaconf import OmegaConf
import optax
from flax.training.train_state import TrainState
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm
import wandb

from conf.config import TrainConfig
from purejaxrl.wrappers import LogWrapper
from puzzlejax.models import ScannedRNN, ActorCriticRNNMulti
from puzzlejax.utils_rl import get_ckpt_dir, init_config
from puzzlescript_jax.env import PJParams, PSObs
import puzzlescript_jax.utils as psutils


MULTI_GAME_PRESETS = {
    # The four games the per-game recurrent agents solved well.
    "win4": ["kettle", "Slidings", "sokodig", "Travelling_salesman"],
    # Increasing-game-count scaling sets (supersets).
    "gen6": ["kettle", "Slidings", "sokodig", "Travelling_salesman",
             "sokoban_match3", "Multi-word_Dictionary_Game"],
    "gen9": ["kettle", "Slidings", "sokodig", "Travelling_salesman",
             "sokoban_match3", "Multi-word_Dictionary_Game",
             "notsnake", "sokoban_basic", "Smother"],
}


def resolve_games(name: str):
    if name in MULTI_GAME_PRESETS:
        return list(MULTI_GAME_PRESETS[name])
    return name.split("+")


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_states: Tuple[Any, ...]
    last_obs: Any
    last_done: jnp.ndarray       # META-episode done (drives RNN-carry reset + GAE)
    trial_count: jnp.ndarray     # completed trials in the current meta-episode
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


def multi_log_callback(metric, game_ids_np, games, csv_path, pbar, stats_bar,
                       n_envs, train_start_time, steps_prev_complete, K=1):
    done_mask = np.asarray(metric["returned_episode"])
    pbar.update(1)
    if done_mask.sum() == 0:
        return
    timesteps = np.asarray(metric["timestep"])[done_mask] * n_envs
    t = int(timesteps[-1]) if timesteps.size else 0
    returns = np.asarray(metric["returned_episode_returns"])[done_mask]
    won = np.asarray(metric["won"])[done_mask].astype(bool)
    # metric leaves are (num_steps, n_envs); broadcast per-env game ids over time.
    gids = np.broadcast_to(game_ids_np[None, :], done_mask.shape)[done_mask]

    per_game_win = {}
    per_game_ret = {}
    payload = {}
    for gi, g in enumerate(games):
        m = gids == gi
        if m.sum() > 0:
            wr = float(won[m].mean())
            rr = float(returns[m].mean())
            per_game_win[g] = wr
            per_game_ret[g] = rr
            payload[f"win_rate/{g}"] = wr
            payload[f"ep_return/{g}"] = rr
    agg_wr = float(won.mean())
    payload["win_rate/_mean"] = agg_wr
    payload["ep_return/_mean"] = float(returns.mean())

    # Few-shot signal: win rate by trial index within the meta-episode. To avoid
    # the game-mix confound (different games sitting at different trial indices
    # in a given window), compute it PER GAME, then average with equal game
    # weight. If the agent adapts in-context, later trials win more often.
    fewshot_str = ""
    if K > 1 and "trial_idx" in metric:
        tidx = np.asarray(metric["trial_idx"])[done_mask]
        per_game_trial = np.full((len(games), K), np.nan)
        for gi in range(len(games)):
            for k in range(K):
                mk = (tidx == k) & (gids == gi)
                if mk.sum() > 0:
                    per_game_trial[gi, k] = won[mk].mean()
        # Equal-weight average over games that have data for that trial index.
        balanced = np.nanmean(per_game_trial, axis=0)
        for k in range(K):
            payload[f"fewshot/trial_{k}_winrate"] = float(balanced[k])
        fewshot_str = " | fewshot(bal) " + " ".join(
            f"t{k}={balanced[k]:.2f}" for k in range(K))
        # Per-game-per-trial detail to a side file: t, then game-major trial-minor.
        fs_path = csv_path.replace("progress.csv", "fewshot.csv")
        with open(fs_path, "a") as f:
            vals = ",".join("" if np.isnan(per_game_trial[gi, k]) else f"{per_game_trial[gi, k]:.3f}"
                            for gi in range(len(games)) for k in range(K))
            f.write(f"{t},{vals}\n")
    wandb.log(payload, step=t)

    fps = (t - steps_prev_complete) / max(timer() - train_start_time, 1e-6)
    desc = (f"  t={t:,} winrate={agg_wr:.2f} FPS={fps:,.0f} | " + " ".join(
        f"{g[:6]}:{per_game_win.get(g, float('nan')):.2f}" for g in games) + fewshot_str)
    stats_bar.set_description_str(desc)

    with open(csv_path, "a") as f:
        f.write(f"{t},{agg_wr}," + ",".join(str(per_game_win.get(g, "")) for g in games)
                + "," + ",".join(str(per_game_ret.get(g, "")) for g in games) + "\n")


def make_train(config: TrainConfig, games, restored_ckpt, checkpoint_manager):
    N = len(games)
    hidden_dim = config.hidden_dims[0]

    # Build per-game envs + params + shapes.
    envs, params_list, shapes = [], [], []
    for g in games:
        e = psutils.init_ps_env(game=g, level_i=-1, max_episode_steps=config.max_episode_steps)
        envs.append(LogWrapper(e))
        params_list.append(PJParams(level=e.get_level(0), level_i=-1))
        shapes.append(tuple(e.observation_space(params_list[-1]).shape))
    Cmax = max(s[0] for s in shapes)
    Hmax = max(s[1] for s in shapes)
    Wmax = max(s[2] for s in shapes)
    print(f"Generalist over {N} games {games}; padded obs ({Cmax},{Hmax},{Wmax})")

    epg = config.n_envs // N
    config.n_envs = epg * N            # make divisible
    n_envs = config.n_envs
    config._num_updates = config.total_timesteps // config.num_steps // n_envs
    config._minibatch_size = n_envs * config.num_steps // config.NUM_MINIBATCHES
    K = int(config.trials_per_meta)
    game_ids = jnp.concatenate([jnp.full((epg,), gi, dtype=jnp.int32) for gi in range(N)])
    game_ids_np = np.asarray(game_ids)
    # What the policy sees: real game id, or a constant (forcing in-context
    # inference) when use_game_id=False.
    game_ids_obs = game_ids if config.use_game_id else jnp.zeros_like(game_ids)
    print(f"meta-RL: trials_per_meta={K}, use_game_id={config.use_game_id}, "
          f"per-trial step budget={config.max_episode_steps}")

    def pad_block(arr, ci, hi, wi):
        return jnp.pad(arr, ((0, 0), (0, Cmax - ci), (0, Hmax - hi), (0, Wmax - wi)))

    def build_obs(multihots):
        stacked = jnp.concatenate(multihots, axis=0)   # (n_envs, Cmax, Hmax, Wmax)
        return PSObs(multihot_level=stacked, flat_obs=game_ids_obs)

    network = ActorCriticRNNMulti(action_dim=5, n_games=N, hidden_dim=hidden_dim,
                                  game_embed_dim=config.game_embed_dim)

    def reset_all(rng):
        multihots, states = [], []
        for gi in range(N):
            rng, _r = jax.random.split(rng)
            keys = jax.random.split(_r, epg)
            obs_g, state_g = jax.vmap(envs[gi].reset, in_axes=(0, None))(keys, params_list[gi])
            ci, hi, wi = shapes[gi]
            multihots.append(pad_block(obs_g.multihot_level, ci, hi, wi))
            states.append(state_g)
        return build_obs(multihots), tuple(states), rng

    def step_all(rng, env_states, action):
        multihots, states, rewards, dones, infos = [], [], [], [], []
        for gi in range(N):
            rng, _r = jax.random.split(rng)
            keys = jax.random.split(_r, epg)
            a_g = action[gi * epg:(gi + 1) * epg]
            obs_g, state_g, r_g, d_g, info_g = jax.vmap(
                envs[gi].step, in_axes=(0, 0, 0, None))(keys, env_states[gi], a_g, params_list[gi])
            ci, hi, wi = shapes[gi]
            multihots.append(pad_block(obs_g.multihot_level, ci, hi, wi))
            states.append(state_g)
            rewards.append(r_g)
            dones.append(d_g)
            infos.append(info_g)
        obs = build_obs(multihots)
        reward = jnp.concatenate(rewards, axis=0)
        done = jnp.concatenate(dones, axis=0)
        info = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *infos)
        return obs, tuple(states), reward, done, info, rng

    def train(rng, config):
        train_start_time = timer()

        rng, _rng = jax.random.split(rng)
        init_obs = PSObs(multihot_level=jnp.zeros((1, 1, Cmax, Hmax, Wmax)),
                         flat_obs=jnp.zeros((1, 1), dtype=jnp.int32))
        init_done = jnp.zeros((1, 1), dtype=bool)
        network_params = network.init(_rng, ScannedRNN.initialize_carry(1, hidden_dim),
                                      (init_obs, init_done))
        tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                         optax.adam(config.lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        rng, _rng = jax.random.split(rng)
        last_obs, env_states, _ = reset_all(_rng)
        hstate = ScannedRNN.initialize_carry(n_envs, hidden_dim)
        last_done = jnp.zeros((n_envs,), dtype=bool)
        trial_count = jnp.zeros((n_envs,), dtype=jnp.int32)

        steps_prev_complete = 0
        runner_state = RunnerState(train_state, env_states, last_obs, last_done,
                                   trial_count, hstate, rng, 0)
        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            config._num_updates = int(
                (config.total_timesteps - steps_prev_complete) // config.num_steps // n_envs)

        pbar = tqdm(total=config._num_updates, desc="Train (rnn-multi)", position=0, dynamic_ncols=True)
        stats_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
        _log_cb = partial(multi_log_callback, game_ids_np=game_ids_np, games=games,
                          csv_path=os.path.join(config._exp_dir, "progress.csv"),
                          pbar=pbar, stats_bar=stats_bar, n_envs=n_envs,
                          train_start_time=train_start_time, steps_prev_complete=steps_prev_complete,
                          K=K)

        def save_checkpoint(runner_state, info):
            ts = info["timestep"][info["returned_episode"]] * n_envs
            if len(ts) > 0:
                t = ts[-1].item()
                latest = checkpoint_manager.latest_step()
                if latest is None or t - latest >= config.ckpt_freq:
                    checkpoint_manager.save(t, args=ocp.args.StandardSave(
                        {'runner_state': runner_state, 'step_i': jnp.array(t, dtype=jnp.int32)}))

        def _update_step(runner_state, unused):
            def _env_step(rs: RunnerState, unused):
                rng = rs.rng
                rng, _rng = jax.random.split(rng)
                ac_in = (jax.tree_util.tree_map(lambda x: x[None], rs.last_obs), rs.last_done[None])
                hstate, pi, value = network.apply(rs.train_state.params, rs.hstate, ac_in)
                action = pi.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(action[None]).squeeze(0)
                value = value.squeeze(0)
                rng, _rng = jax.random.split(rng)
                obs, env_states, reward, done, info, _ = step_all(_rng, rs.env_states, action)
                # Trial done = env done (win or per-trial step budget). The level
                # resets each trial (env auto-reset), but the RNN carry only
                # resets at META-episode boundaries (every K-th trial), so memory
                # persists across trials for in-context (few-shot) adaptation.
                new_tc = rs.trial_count + done.astype(jnp.int32)
                meta_done = jnp.logical_and(done, (new_tc % K) == 0)
                trial_count_next = jnp.where(meta_done, jnp.int32(0), new_tc)
                info = {**info, "trial_idx": rs.trial_count}
                transition = Transition(rs.last_done, action, value, reward, log_prob, rs.last_obs, info)
                rs = RunnerState(rs.train_state, env_states, obs, meta_done,
                                 trial_count_next, hstate, rng, rs.update_i)
                return rs, transition

            init_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            ac_in = (jax.tree_util.tree_map(lambda x: x[None], runner_state.last_obs),
                     runner_state.last_done[None])
            _, _, last_val = network.apply(runner_state.train_state.params, runner_state.hstate, ac_in)
            last_val = last_val.squeeze(0)
            last_done = runner_state.last_done

            def _calculate_gae(traj_batch, last_val, last_done):
                def _adv(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.GAMMA * next_value * (1 - next_done) - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(
                    _adv, (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params, init_hstate.squeeze(0), (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets)).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1 - config.CLIP_EPS, 1 + config.CLIP_EPS) * gae).mean()
                        entropy = pi.entropy().mean()
                        return loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy, None

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, _), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, n_envs)
                batch = (init_hstate[None, :], traj_batch, advantages, targets)
                shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config.NUM_MINIBATCHES, -1] + list(x.shape[2:])), 1, 0),
                    shuffled)
                train_state, loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, init_hstate, traj_batch, advantages, targets, rng), loss

            rng = runner_state.rng
            update_state = (runner_state.train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            train_state = update_state[0]
            rng = update_state[-1]
            metric = traj_batch.info

            jax.debug.callback(save_checkpoint, runner_state, metric)
            jax.debug.callback(_log_cb, metric)

            runner_state = RunnerState(train_state, runner_state.env_states, runner_state.last_obs,
                                       runner_state.last_done, runner_state.trial_count,
                                       runner_state.hstate, rng, runner_state.update_i + 1)
            return runner_state, None

        runner_state, _ = jax.lax.scan(_update_step, runner_state, None, config._num_updates)
        return {"runner_state": runner_state}

    return lambda rng: train(rng, config), network, (Cmax, Hmax, Wmax), n_envs, game_ids


@hydra.main(version_base="1.3", config_path='../conf', config_name='train')
def main(config: TrainConfig):
    logging.getLogger().setLevel(logging.WARNING)
    games = resolve_games(config.game)
    config.model = 'rnn_multi'
    config = init_config(config)
    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config._exp_dir
    print(f'Generalist recurrent run logged at {exp_dir}\n')
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_dir = os.path.abspath(get_ckpt_dir(config))
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True))

    run = wandb.init(project=getattr(config, "wandb_project", "puzzlejax_ppo"),
                     config=OmegaConf.to_container(config),
                     mode=getattr(config, "wandb_mode", "online"), dir=exp_dir)

    with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
        f.write("timestep,win_rate_mean," + ",".join(f"win/{g}" for g in games)
                + "," + ",".join(f"ret/{g}" for g in games) + "\n")

    train_fn, network, pad_shape, n_envs, game_ids = make_train(config, games, None, checkpoint_manager)
    train_jit = jax.jit(train_fn)
    out = train_jit(rng)
    jax.block_until_ready(out)


if __name__ == "__main__":
    main()
