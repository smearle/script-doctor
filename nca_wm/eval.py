"""Evaluation harness for the NCA world model.

Autoregressive multi-step rollout evaluation: single-rollout and batched
(numpy and jitted-jax) implementations, an implementation benchmark, and the
multi-game aggregator that scores a checkpoint against random / BFS / A*
rollouts and writes the run scorecard. Imports the shared inference helpers,
the model classes, and the data-collection cache/solution loaders, but never
train.py.
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import wandb

from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from nca_wm.models import NCAWorldModel, ConditionalNCAWorldModel
from nca_wm.data_collection import (
    _cache_dir, _load_npz_dict, _save_npz_dict, _enabled_action_count,
    _rollout_history, _solution_from_sol_dir, _solution_from_transitions_cache,
)
from nca_wm.inference import make_apply_fn, _pad_state_for_model

N_ACTIONS = 5

# Module-level cache of JIT'd scan functions, keyed by (id(model), conditional).
# Hoisting these out of `_run_eval_rollouts_jax` is important: defining them
# inside the function would create fresh closures (and fresh JIT caches) on
# every call — so a Python loop over (game, level, mode) would recompile each
# time and JAX would be slower than the unbatched path.
_JAX_EVAL_CACHE: dict = {}


def evaluate_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    level_i: int = 0,
    n_episodes: int = 10,
    max_steps: int = 50,
    save_dir: str | None = None,
    history: int = 0,
):
    """Roll out the world model alongside the real env and measure divergence.

    When ``history > 0``, the model is fed its own recently unrolled
    (state, action) pairs as history — the rollout trajectory itself is the
    natural source, mirroring training-time semantics. Early steps with fewer
    than ``history`` predecessors get zero-padded (masked) history slots.
    """
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    apply_fn = make_apply_fn(model)

    n_act = _enabled_action_count(json_str)
    all_l1_errors = []
    for ep_i in range(n_episodes):
        real_obs, _ = env.reset()
        pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
        ep_errors = []
        hist_buf: list[tuple[np.ndarray, int]] = []  # (state (C,H,W), action)

        for t in range(max_steps):
            action = np.random.randint(n_act)
            a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
            hs, ha = _rollout_history(hist_buf, history, pred_state[0].shape)

            # World model prediction
            logits, _win_logit, _sprite_logits = apply_fn(
                params, pred_state, a_oh, hist_states=hs, hist_actions=ha)
            if history > 0:
                # Record the (state, action) we just predicted from.
                hist_buf.append((np.asarray(pred_state[0]), action))
            pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

            # Real env step
            real_obs, _, done, truncated, _ = env.step(action)
            real = jnp.array(real_obs[None], dtype=jnp.float32)

            l1 = float(jnp.abs(pred_state - real).sum())
            ep_errors.append(l1)
            if done or truncated:
                break

        all_l1_errors.append(ep_errors)

    # Report per-step average divergence
    max_len = max(len(e) for e in all_l1_errors)
    padded = np.full((n_episodes, max_len), np.nan)
    for i, e in enumerate(all_l1_errors):
        padded[i, :len(e)] = e
    mean_per_step = np.nanmean(padded, axis=0)
    print(f"Eval ({n_episodes} eps): step-1 L1={mean_per_step[0]:.1f}, "
          f"step-5 L1={mean_per_step[min(4, len(mean_per_step)-1)]:.1f}, "
          f"step-20 L1={mean_per_step[min(19, len(mean_per_step)-1)]:.1f}")

    if save_dir:
        np.savez(os.path.join(save_dir, "eval_divergence.npz"),
                 mean_per_step=mean_per_step, all_errors=padded)

    return mean_per_step


def _run_eval_rollout(
    apply_fn, params, json_str: str,
    level_i: int, n_objs: int,
    max_C: int, max_H: int, max_W: int,
    actions: list[int] | None = None,
    max_steps: int = 50,
    game_tokens: np.ndarray | None = None,
    game_mask: np.ndarray | None = None,
    teacher_forced: bool = False,
) -> dict:
    """Run a single eval rollout and return per-step metrics.

    Autoregressive (default): model feeds its own prediction back each step.
    Teacher-forced: model is fed the real (env) state each step. Isolates
    single-step prediction error from compounding drift.

    Returns dict with:
        wrong_tiles: (T,) int — per-bit mismatches per step
        wrong_cells: (T,) int — cells where ANY object-channel bit is wrong
        tile_error_rate: (T,) float — wrong_tiles / total_bits per step
        cell_error_rate: (T,) float — wrong_cells / (H*W) per step
        first_div_step: int — first t with wrong_cells > 0 (-1 if never)
        total_tiles: int — n_objs * H * W
        total_cells: int — H * W
    """
    conditional = game_tokens is not None
    if conditional:
        gt = jnp.array(game_tokens[None])  # (1, S)
        gm = jnp.array(game_mask[None])    # (1, S)

    env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                             max_episode_steps=max_steps if actions is None else len(actions))
    real_obs, _ = env.reset()
    _, H, W = real_obs.shape
    total_tiles = n_objs * H * W
    total_cells = H * W
    pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)

    n_steps = len(actions) if actions else max_steps
    n_act = _enabled_action_count(json_str)
    wrong_tiles = []
    wrong_cells = []
    first_div = -1
    for t in range(n_steps):
        action = actions[t] if actions else np.random.randint(n_act)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        if conditional:
            logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        real_obs, _, done, truncated, _ = env.step(action)

        pred_binary = np.array(
            pred_next[0, :n_objs, :H, :W] > 0.5, dtype=np.uint8,
        )
        mismatch = (pred_binary != real_obs)
        n_wrong_bits = int(mismatch.sum())
        n_wrong_cells = int(mismatch.any(axis=0).sum())
        wrong_tiles.append(n_wrong_bits)
        wrong_cells.append(n_wrong_cells)
        if first_div == -1 and n_wrong_cells > 0:
            first_div = t

        # Next input: model's own prediction (autoregressive) or re-padded real
        # state (teacher-forced). For AR, zero outside the level's actual
        # (n_objs, H, W) extent so off-level binarized predictions don't leak
        # in as OOD nonzero input — training inputs always have those regions
        # exactly zero (see _pad_state_for_model + bucket loader).
        if teacher_forced:
            pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)
        else:
            clean = jnp.zeros_like(pred_next)
            clean = clean.at[:, :n_objs, :H, :W].set(pred_next[:, :n_objs, :H, :W])
            pred_state = clean

        if done or truncated:
            break

    wrong_tiles = np.array(wrong_tiles)
    wrong_cells = np.array(wrong_cells)
    return {
        "wrong_tiles": wrong_tiles,
        "wrong_cells": wrong_cells,
        "tile_error_rate": wrong_tiles / total_tiles,
        "cell_error_rate": wrong_cells / total_cells,
        "first_div_step": first_div,
        "total_tiles": total_tiles,
        "total_cells": total_cells,
    }


def _run_eval_rollouts_batched(
    apply_fn, params, json_str: str,
    level_i: int, n_objs: int,
    max_C: int, max_H: int, max_W: int,
    n_episodes: int,
    max_steps: int = 50,
    game_tokens: np.ndarray | None = None,
    game_mask: np.ndarray | None = None,
    teacher_forced: bool = False,
    rng_seed: int = 0,
    actions_2d: np.ndarray | None = None,
    history: int = 0,
) -> dict:
    """Batched random-rollout eval — runs ``n_episodes`` random rollouts in
    parallel through one JIT'd forward per step (batch=n_episodes). Same
    semantics as ``_run_eval_rollout`` with ``actions=None``, just stacked.

    When ``history > 0`` each episode carries a rolling buffer of its own
    last ``history`` (predicted state, action) pairs, fed as history — the
    faithful rollout-time analogue of training, instead of the zero history
    the apply_fn safety net would otherwise inject.

    Returns:
        wrong_tiles_grid: (n_episodes, T) float — NaN past per-ep termination
        wrong_cells_grid: (n_episodes, T) float
        first_div: (n_episodes,) int — first divergent step (-1 if never)
        per_ep_length: (n_episodes,) int — recorded steps per episode
        total_tiles, total_cells: ints
    """
    conditional = game_tokens is not None
    if conditional:
        gt_b = jnp.broadcast_to(jnp.array(game_tokens[None]),
                                (n_episodes, game_tokens.shape[0]))
        gm_b = jnp.broadcast_to(jnp.array(game_mask[None]),
                                (n_episodes, game_mask.shape[0]))

    envs = []
    initial_obs = None
    for _ in range(n_episodes):
        env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                                 max_episode_steps=max_steps)
        obs, _ = env.reset()
        if initial_obs is None:
            initial_obs = obs
        envs.append(env)
    _, H, W = initial_obs.shape
    total_tiles = n_objs * H * W
    total_cells = H * W

    pad0 = _pad_state_for_model(initial_obs, max_C, max_H, max_W)  # (1, C, H', W')
    pred_states = jnp.broadcast_to(pad0, (n_episodes,) + pad0.shape[1:])

    if actions_2d is not None:
        # Caller-provided (n_eps, max_steps) layout — transpose to (T, n_eps).
        actions_per_step = np.asarray(actions_2d, dtype=np.int32).T
    else:
        rng = np.random.default_rng(rng_seed)
        actions_per_step = rng.integers(0, _enabled_action_count(json_str),
                                        size=(max_steps, n_episodes), dtype=np.int32)
    eye = np.eye(N_ACTIONS, dtype=np.float32)

    wrong_tiles_grid = np.full((n_episodes, max_steps), np.nan, dtype=np.float64)
    wrong_cells_grid = np.full((n_episodes, max_steps), np.nan, dtype=np.float64)
    first_div = np.full(n_episodes, -1, dtype=np.int64)
    per_ep_length = np.zeros(n_episodes, dtype=np.int32)
    done_mask = np.zeros(n_episodes, dtype=bool)

    # Per-episode rolling history buffers (oldest at col 0, newest at col -1).
    hist_s_roll = (np.zeros((n_episodes, history, max_C, max_H, max_W),
                            dtype=np.float32) if history > 0 else None)
    hist_a_roll = (np.zeros((n_episodes, history), dtype=np.int32)
                   if history > 0 else None)

    for t in range(max_steps):
        was_alive = ~done_mask.copy()
        if not was_alive.any():
            break

        a_oh = jnp.array(eye[actions_per_step[t]])  # (n_eps, N_ACTIONS)
        hkw = {}
        if history > 0:
            hkw = dict(hist_states=jnp.array(hist_s_roll),
                       hist_actions=jnp.array(hist_a_roll))
        if conditional:
            logits, _, _ = apply_fn(params, pred_states, a_oh, gt_b, gm_b, **hkw)
        else:
            logits, _, _ = apply_fn(params, pred_states, a_oh, **hkw)
        # Roll the (state, action) we predicted from into the history buffer.
        if history > 0:
            prev = np.asarray(pred_states)
            hist_s_roll[:, :-1] = hist_s_roll[:, 1:]
            hist_s_roll[:, -1] = prev
            hist_a_roll[:, :-1] = hist_a_roll[:, 1:]
            hist_a_roll[:, -1] = actions_per_step[t]
        pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        real_obs_batch = np.zeros((n_episodes, n_objs, H, W), dtype=np.uint8)
        for i in range(n_episodes):
            if not was_alive[i]:
                continue
            obs, _, done, trunc, _ = envs[i].step(int(actions_per_step[t, i]))
            real_obs_batch[i] = obs
            if done or trunc:
                done_mask[i] = True

        pred_binary = np.array(pred_next[:, :n_objs, :H, :W] > 0.5, dtype=np.uint8)
        mismatch = (pred_binary != real_obs_batch)
        n_wrong_bits = mismatch.sum(axis=(1, 2, 3))
        n_wrong_cells = mismatch.any(axis=1).sum(axis=(1, 2))
        for i in range(n_episodes):
            if was_alive[i]:
                wrong_tiles_grid[i, t] = n_wrong_bits[i]
                wrong_cells_grid[i, t] = n_wrong_cells[i]
                per_ep_length[i] = t + 1
                if first_div[i] == -1 and n_wrong_cells[i] > 0:
                    first_div[i] = t

        if teacher_forced:
            new_states = np.zeros((n_episodes, max_C, max_H, max_W), dtype=np.float32)
            new_states[:, :n_objs, :H, :W] = real_obs_batch
            pred_states = jnp.array(new_states)
        else:
            clean = jnp.zeros_like(pred_next)
            clean = clean.at[:, :n_objs, :H, :W].set(pred_next[:, :n_objs, :H, :W])
            pred_states = clean

    max_len = int(per_ep_length.max()) if per_ep_length.max() > 0 else 0
    return {
        "wrong_tiles_grid": wrong_tiles_grid[:, :max_len],
        "wrong_cells_grid": wrong_cells_grid[:, :max_len],
        "first_div": first_div,
        "per_ep_length": per_ep_length,
        "total_tiles": total_tiles,
        "total_cells": total_cells,
    }


def _get_jax_eval_fns(model, conditional: bool):
    key = (id(model), conditional)
    if key in _JAX_EVAL_CACHE:
        return _JAX_EVAL_CACHE[key]

    _hist_k = int(getattr(model, "history", 0))

    def _trimmed_apply(p, st, a, *cond):
        # The scan rollout does not carry a real history buffer; feed zero
        # (masked) history when the model expects history channels so the
        # embed width matches. This makes the scan's rollout-divergence metric
        # a pessimistic, context-free estimate for history>0 models — the
        # primary held-out change_err metric (real history) is unaffected, and
        # the single-game AR rollout (evaluate_world_model) uses real history.
        hkw = {}
        if _hist_k > 0:
            hkw = dict(
                hist_states=jnp.zeros((st.shape[0], _hist_k) + tuple(st.shape[1:]),
                                      st.dtype),
                hist_actions=jnp.zeros((st.shape[0], _hist_k), jnp.int32),
            )
        out = model.apply(p, st, a, *cond, **hkw)
        return out[0]  # logits

    if conditional:
        @jax.jit
        def _ar_scan(p, init, a_T, real_T, vmask, cmask, gt_b, gm_b):
            def body(carry, inp):
                a_oh, real_next = inp
                logits = _trimmed_apply(p, carry, a_oh, gt_b, gm_b)
                pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                diff = (pred_next != real_next).astype(jnp.float32) * vmask
                wb = diff.sum(axis=(1, 2, 3))
                cd = (diff.sum(axis=1) > 0).astype(jnp.float32)
                wc = (cd * cmask).sum(axis=(1, 2))
                return pred_next * vmask, (wb, wc)
            _, outs = jax.lax.scan(body, init, (a_T, real_T))
            return outs

        @jax.jit
        def _tf_scan(p, st_T, a_T, real_T, vmask, cmask, gt_b, gm_b):
            def body(_, inp):
                states, a_oh, real_next = inp
                logits = _trimmed_apply(p, states, a_oh, gt_b, gm_b)
                pred = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                diff = (pred != real_next).astype(jnp.float32) * vmask
                wb = diff.sum(axis=(1, 2, 3))
                cd = (diff.sum(axis=1) > 0).astype(jnp.float32)
                wc = (cd * cmask).sum(axis=(1, 2))
                return None, (wb, wc)
            _, outs = jax.lax.scan(body, None, (st_T, a_T, real_T))
            return outs
    else:
        @jax.jit
        def _ar_scan(p, init, a_T, real_T, vmask, cmask):
            def body(carry, inp):
                a_oh, real_next = inp
                logits = _trimmed_apply(p, carry, a_oh)
                pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                diff = (pred_next != real_next).astype(jnp.float32) * vmask
                wb = diff.sum(axis=(1, 2, 3))
                cd = (diff.sum(axis=1) > 0).astype(jnp.float32)
                wc = (cd * cmask).sum(axis=(1, 2))
                return pred_next * vmask, (wb, wc)
            _, outs = jax.lax.scan(body, init, (a_T, real_T))
            return outs

        @jax.jit
        def _tf_scan(p, st_T, a_T, real_T, vmask, cmask):
            def body(_, inp):
                states, a_oh, real_next = inp
                logits = _trimmed_apply(p, states, a_oh)
                pred = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                diff = (pred != real_next).astype(jnp.float32) * vmask
                wb = diff.sum(axis=(1, 2, 3))
                cd = (diff.sum(axis=1) > 0).astype(jnp.float32)
                wc = (cd * cmask).sum(axis=(1, 2))
                return None, (wb, wc)
            _, outs = jax.lax.scan(body, None, (st_T, a_T, real_T))
            return outs

    fns = {"ar": _ar_scan, "tf": _tf_scan}
    _JAX_EVAL_CACHE[key] = fns
    return fns


def _run_eval_rollouts_jax(
    model, params, json_str: str,
    level_i: int, n_objs: int,
    max_C: int, max_H: int, max_W: int,
    n_episodes: int,
    max_steps: int = 50,
    game_tokens: np.ndarray | None = None,
    game_mask: np.ndarray | None = None,
    teacher_forced: bool = False,
    rng_seed: int = 0,
    actions_2d: np.ndarray | None = None,
    return_both: bool = False,
) -> dict:
    """Fully JAX-side eval: pre-roll the C++ env once, then run the entire
    model rollout as a single JIT'd ``lax.scan`` over time. Eliminates
    per-step Python/JAX dispatch.

    Both AR and TF use scan over time, batching only across episodes — so
    peak memory is the same (n_episodes per-step), no OOM at large max_steps.

    With ``return_both=True`` the SAME pre-rolled real trajectory and SAME
    action sequence feed both the autoregressive and teacher-forced scans, so
    the two sets of metrics describe the identical random rollout (and the
    env is pre-rolled only once). Returned keys are then prefixed ``ar_`` /
    ``tf_`` (``ar_wrong_tiles_grid``, ``tf_first_div``, …). The action grid is
    always returned under ``actions_2d`` for reproducible re-rendering.

    Returns the same dict shape as ``_run_eval_rollouts_batched`` (single
    mode) or the prefixed form (``return_both``).
    """
    conditional = game_tokens is not None

    # 1. Generate or accept actions.
    if actions_2d is not None:
        actions_2d = np.asarray(actions_2d, dtype=np.int32)
    else:
        rng = np.random.default_rng(rng_seed)
        actions_2d = rng.integers(0, _enabled_action_count(json_str),
                                  size=(n_episodes, max_steps), dtype=np.int32)

    # 2. Pre-roll real envs in C++ (sequential but fast).
    real_obs_traj = None
    per_ep_length = np.zeros(n_episodes, dtype=np.int32)
    H = W = 0
    for ep_i in range(n_episodes):
        env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                                 max_episode_steps=max_steps)
        obs, _ = env.reset()
        if real_obs_traj is None:
            _, H, W = obs.shape
            real_obs_traj = np.zeros((n_episodes, max_steps + 1, n_objs, H, W),
                                     dtype=np.uint8)
        real_obs_traj[ep_i, 0] = obs
        for t in range(max_steps):
            obs, _, done, trunc, _ = env.step(int(actions_2d[ep_i, t]))
            real_obs_traj[ep_i, t + 1] = obs
            per_ep_length[ep_i] = t + 1
            if done or trunc:
                break

    total_tiles = n_objs * H * W
    total_cells = H * W

    # 3. Pad real trajectory to bucket dims and build masks.
    real_padded = np.zeros((n_episodes, max_steps + 1, max_C, max_H, max_W),
                           dtype=np.float32)
    real_padded[:, :, :n_objs, :H, :W] = real_obs_traj
    real_padded_jax = jnp.array(real_padded)

    valid_mask = np.zeros((max_C, max_H, max_W), dtype=np.float32)
    valid_mask[:n_objs, :H, :W] = 1.0
    valid_mask_jax = jnp.array(valid_mask)
    cell_valid_mask = np.zeros((max_H, max_W), dtype=np.float32)
    cell_valid_mask[:H, :W] = 1.0
    cell_valid_mask_jax = jnp.array(cell_valid_mask)

    # 4. Action one-hots.
    eye = np.eye(N_ACTIONS, dtype=np.float32)
    a_oh_traj = jnp.array(eye[actions_2d])  # (n_eps, max_steps, N_ACTIONS)

    # 5. Conditional broadcast.
    if conditional:
        gt_b = jnp.broadcast_to(jnp.array(game_tokens[None]),
                                (n_episodes, game_tokens.shape[0]))
        gm_b = jnp.broadcast_to(jnp.array(game_mask[None]),
                                (n_episodes, game_mask.shape[0]))

    fns = _get_jax_eval_fns(model, conditional)
    a_oh_T = jnp.transpose(a_oh_traj, (1, 0, 2))                     # (T, n_eps, A)
    real_next_T = jnp.transpose(real_padded_jax[:, 1:],
                                (1, 0, 2, 3, 4))                     # (T, n_eps, C, H, W)
    states_in_T = jnp.transpose(real_padded_jax[:, :max_steps],
                                (1, 0, 2, 3, 4))                     # (T, n_eps, C, H, W)
    init_state = real_padded_jax[:, 0]                              # (n_eps, C, H, W)
    cond_args = (gt_b, gm_b) if conditional else ()

    def _run_tf():
        return fns["tf"](params, states_in_T, a_oh_T, real_next_T,
                         valid_mask_jax, cell_valid_mask_jax, *cond_args)

    def _run_ar():
        return fns["ar"](params, init_state, a_oh_T, real_next_T,
                         valid_mask_jax, cell_valid_mask_jax, *cond_args)

    max_len = int(per_ep_length.max()) if per_ep_length.max() > 0 else 0

    def _postprocess(wb_T, wc_T):
        # (T, n_eps) -> (n_eps, T), mask past per-ep termination, first-div.
        wb = np.asarray(jnp.transpose(wb_T, (1, 0)))
        wc = np.asarray(jnp.transpose(wc_T, (1, 0)))
        grid_bits = np.full((n_episodes, max_steps), np.nan, dtype=np.float64)
        grid_cells = np.full((n_episodes, max_steps), np.nan, dtype=np.float64)
        first_div = np.full(n_episodes, -1, dtype=np.int64)
        for i in range(n_episodes):
            L = int(per_ep_length[i])
            grid_bits[i, :L] = wb[i, :L]
            grid_cells[i, :L] = wc[i, :L]
            for t in range(L):
                if wc[i, t] > 0:
                    first_div[i] = t
                    break
        return {
            "wrong_tiles_grid": grid_bits[:, :max_len],
            "wrong_cells_grid": grid_cells[:, :max_len],
            "first_div": first_div,
        }

    common = {
        "per_ep_length": per_ep_length,
        "total_tiles": total_tiles,
        "total_cells": total_cells,
        "actions_2d": actions_2d[:, :max_len] if max_len > 0 else actions_2d,
    }

    if return_both:
        # Same pre-rolled trajectory + same actions feed BOTH scans, so AR and
        # TF describe the identical rollout. With shared actions the per-episode
        # invariant is exact: if TF is correct at every step, AR re-derives the
        # same states it would be fed under TF, so AR is correct too — any AR
        # error therefore coincides with a nonzero 1-step (TF) error.
        out = dict(common)
        for prefix, res in (("ar", _postprocess(*_run_ar())),
                            ("tf", _postprocess(*_run_tf()))):
            for k, v in res.items():
                out[f"{prefix}_{k}"] = v
        return out

    out = _postprocess(*(_run_tf() if teacher_forced else _run_ar()))
    out.update(common)
    return out


def _benchmark_eval_impls(model, params, game_infos,
                          n_episodes: int = 10, max_steps: int = 50,
                          rng_seed: int = 0):
    """Time the three random-rollout eval implementations on the first
    (game, level=0) using a shared action sequence so metrics agree.

    Reports wall time per implementation per mode (AR + teacher-forced).
    """
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    from nca_wm.baselines import CNNWorldModel, UNetWorldModel, ViTWorldModel
    conditional = isinstance(
        model,
        (ConditionalNCAWorldModel, RuleAttnNCAWorldModel,
         CNNWorldModel, UNetWorldModel, ViTWorldModel),
    )
    info = game_infos[0]
    name = info["name"]
    json_str = info["json_str"]
    n_objs = info["n_objs"]
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    apply_fn = make_apply_fn(model)

    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)
        tids = info.get("token_ids", [])
        gt = np.zeros(max_tok_len, dtype=np.int32)
        gm = np.zeros(max_tok_len, dtype=np.bool_)
        gt[:len(tids)] = tids
        gm[:len(tids)] = True
        cond_kwargs = {"game_tokens": gt, "game_mask": gm}
    else:
        cond_kwargs = {}

    rng = np.random.default_rng(rng_seed)
    actions_2d = rng.integers(0, _enabled_action_count(json_str),
                              size=(n_episodes, max_steps), dtype=np.int32)

    print(f"\n=== Benchmark on {name} L0 "
          f"(n_eps={n_episodes}, max_steps={max_steps}, n_objs={n_objs}, "
          f"bucket={max_C}x{max_H}x{max_W}) ===")

    def _run_seq(tf):
        all_bits, all_cells, first_divs = [], [], []
        for ep_i in range(n_episodes):
            r = _run_eval_rollout(
                apply_fn, params, json_str, level_i=0, n_objs=n_objs,
                max_C=max_C, max_H=max_H, max_W=max_W,
                actions=actions_2d[ep_i].tolist(),
                max_steps=max_steps,
                teacher_forced=tf,
                **cond_kwargs,
            )
            all_bits.append(r["wrong_tiles"])
            all_cells.append(r["wrong_cells"])
            first_divs.append(r["first_div_step"])
        return all_bits, all_cells, first_divs, r["total_tiles"]

    def _run_batched(tf):
        return _run_eval_rollouts_batched(
            apply_fn, params, json_str, level_i=0, n_objs=n_objs,
            max_C=max_C, max_H=max_H, max_W=max_W,
            n_episodes=n_episodes, max_steps=max_steps,
            teacher_forced=tf, actions_2d=actions_2d,
            **cond_kwargs,
        )

    def _run_jax(tf):
        return _run_eval_rollouts_jax(
            model, params, json_str, level_i=0, n_objs=n_objs,
            max_C=max_C, max_H=max_H, max_W=max_W,
            n_episodes=n_episodes, max_steps=max_steps,
            teacher_forced=tf, actions_2d=actions_2d,
            **cond_kwargs,
        )

    def _block_until_ready(out):
        # Force any deferred JAX computation to complete before timing stops.
        try:
            jax.block_until_ready(out)
        except Exception:
            pass
        return out

    def _time(fn, tf, n_warm=1, n_runs=3):
        for _ in range(n_warm):
            _block_until_ready(fn(tf))
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = fn(tf)
            _block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts)), float(np.std(ts)), out

    results = {}
    for mode_label, tf in [("AR random", False), ("teacher-forced", True)]:
        seq_t, seq_s, seq_out = _time(_run_seq, tf)
        bat_t, bat_s, bat_out = _time(_run_batched, tf)
        jax_t, jax_s, jax_out = _time(_run_jax, tf)
        results[mode_label] = {
            "seq": (seq_t, seq_s),
            "batched": (bat_t, bat_s),
            "jax": (jax_t, jax_s),
        }

        # Sanity-check that all three agree on metrics (within tiny float noise).
        seq_bits = seq_out[0]
        seq_total_tiles = seq_out[3]
        max_len_seq = max(len(b) for b in seq_bits)
        seq_grid = np.full((n_episodes, max_len_seq), np.nan)
        for i, b in enumerate(seq_bits):
            seq_grid[i, :len(b)] = b
        bat_grid = bat_out["wrong_tiles_grid"]
        jax_grid = jax_out["wrong_tiles_grid"]
        T = min(seq_grid.shape[1], bat_grid.shape[1], jax_grid.shape[1])
        # Compare element-wise on overlapping shape
        diff_seq_bat = np.nanmax(np.abs(
            np.nan_to_num(seq_grid[:, :T]) - np.nan_to_num(bat_grid[:, :T])))
        diff_seq_jax = np.nanmax(np.abs(
            np.nan_to_num(seq_grid[:, :T]) - np.nan_to_num(jax_grid[:, :T])))
        agree_seq_bat = "OK" if diff_seq_bat < 1e-3 else f"DIFF max={diff_seq_bat}"
        agree_seq_jax = "OK" if diff_seq_jax < 1e-3 else f"DIFF max={diff_seq_jax}"

        print(f"\n[{mode_label}]")
        print(f"  per-episode loop:  {seq_t*1000:7.1f} ms  (±{seq_s*1000:.1f})")
        print(f"  batched:           {bat_t*1000:7.1f} ms  (±{bat_s*1000:.1f})  "
              f"[{seq_t/bat_t:5.1f}x]  metric vs seq: {agree_seq_bat}")
        print(f"  JAX-scanned:       {jax_t*1000:7.1f} ms  (±{jax_s*1000:.1f})  "
              f"[{seq_t/jax_t:5.1f}x]  metric vs seq: {agree_seq_jax}")

    return results


def evaluate_multigame(
    model: NCAWorldModel,
    params,
    game_infos: list[dict],
    ps_parser=None,
    n_random_episodes: int = 10,
    max_steps: int = 50,
    search_algos: list[str] = ("bfs", "astar"),
    search_n_steps: int = 100_000,
    search_timeout_ms: int = -1,
    save_dir: str | None = None,
):
    """Evaluate per game, per level, per rollout type (random + search).

    Reports tile discrepancy counts and error rates.
    """
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    from nca_wm.baselines import CNNWorldModel, UNetWorldModel, ViTWorldModel
    conditional = isinstance(
        model,
        (ConditionalNCAWorldModel, RuleAttnNCAWorldModel,
         CNNWorldModel, UNetWorldModel, ViTWorldModel),
    )
    apply_fn = make_apply_fn(model)

    # Prepare padded token arrays for conditional eval
    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)

    def _get_token_data(info):
        if not conditional:
            return {}, {}
        tids = info.get("token_ids", [])
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        return padded, mask

    results = {}  # results[game][level_i][rollout_type] = dict of metrics

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        game_tokens, game_mask = _get_token_data(info)
        cond_kwargs = {}
        if conditional:
            cond_kwargs = {"game_tokens": game_tokens, "game_mask": game_mask}
        game_results = {}

        for level_i in range(n_levels):
            level_results = {}

            # --- Random rollouts (autoregressive + teacher-forced) ---
            # ONE pre-roll of the C++ env feeds BOTH scans (return_both), so
            # AR ("random") and TF ("random_tf") report on the identical
            # action sequence and real trajectory — directly comparable
            # per-step/per-episode, and the env is rolled only once. The
            # rollout itself is a single JIT'd lax.scan; see
            # _benchmark_eval_impls for the per-impl timing comparison.
            r_both = _run_eval_rollouts_jax(
                model, params, json_str, level_i, n_objs,
                max_C, max_H, max_W,
                n_episodes=n_random_episodes,
                max_steps=max_steps,
                return_both=True, **cond_kwargs,
            )
            for mode_name, prefix in [("random", "ar"), ("random_tf", "tf")]:
                bits_p = r_both[f"{prefix}_wrong_tiles_grid"]
                cells_p = r_both[f"{prefix}_wrong_cells_grid"]
                max_len = bits_p.shape[1]
                mean_bits = (np.nanmean(bits_p, axis=0)
                             if max_len > 0 else np.zeros(0))
                mean_cells = (np.nanmean(cells_p, axis=0)
                              if max_len > 0 else np.zeros(0))
                # First-divergence: treat -1 (no divergence) as max_len (best case)
                fd = np.array([max_len if x < 0 else x
                               for x in r_both[f"{prefix}_first_div"]])
                level_results[mode_name] = {
                    "mean_error_rate": mean_bits / r_both["total_tiles"],
                    "mean_cell_error_rate": mean_cells / r_both["total_cells"],
                    "mean_wrong_tiles": mean_bits,
                    "mean_wrong_cells": mean_cells,
                    "mean_first_div": float(fd.mean()),
                    "total_tiles": r_both["total_tiles"],
                    "total_cells": r_both["total_cells"],
                }

            # --- Search rollouts ---
            backend_search = CppPuzzleScriptBackend()
            backend_search.load_from_json(json_str)
            for algo in search_algos:
                # Try cache first, then training-transitions extraction, then
                # actually re-run search. Both shortcuts yield action IDs in the
                # C++-backend convention, which is what `_run_eval_rollout`
                # consumes — so they're drop-in interchangeable.
                cache_path = os.path.join(
                    _cache_dir(name, level_i),
                    f"search_{algo}_{search_n_steps}_{search_timeout_ms}.npz"
                )
                cached = _load_npz_dict(cache_path)
                sol_actions = None
                cache_valid = False
                if cached is not None and len(cached["actions"]) > 0:
                    source_algo = str(cached.get("source_algo", ""))
                    cache_valid = source_algo == algo
                if cache_valid:
                    sol_actions = cached["actions"].tolist()
                source_kind = "search_cache" if sol_actions is not None else ""
                if sol_actions is None:
                    # Reuse the winning trajectory the training-time collector
                    # already explored. Saves up to `search_timeout_ms` of
                    # wall-clock per (game, level, algo) on hard games where
                    # search would otherwise time out at eval.
                    sol_actions = _solution_from_transitions_cache(
                        name, level_i, search_algo=algo,
                    )
                    source_kind = "transitions_cache" if sol_actions is not None else ""
                if sol_actions is None:
                    # Pre-computed solutions from prior search runs. Both
                    # cpp_sols and js_sols store the C++-backend action IDs eval
                    # uses (read as-is — see _solution_from_sol_dir).
                    sol_actions = _solution_from_sol_dir(
                        os.path.join(_REPO_ROOT, "data", "cpp_sols"),
                        name, level_i, algos=(algo,),
                    )
                    source_kind = "cpp_sols" if sol_actions is not None else ""
                if sol_actions is None:
                    sol_actions = _solution_from_sol_dir(
                        os.path.join(_REPO_ROOT, "data", "js_sols"),
                        name, level_i, algos=(algo,),
                    )
                    source_kind = "js_sols" if sol_actions is not None else ""
                if sol_actions is None:
                    try:
                        backend_search.load_level("", level_i)
                        result = backend_search.run_search(
                            algo, game_text="", level_i=level_i,
                            n_steps=search_n_steps, timeout_ms=search_timeout_ms,
                        )
                        if not result.actions:
                            continue
                        sol_actions = list(result.actions)
                        source_kind = "live_search"
                    except Exception:
                        continue
                # Persist whatever we ended up with so the next eval run
                # (this run or any other model trained on the same game) is
                # entirely search-free for this (algo, budget, timeout).
                if not cache_valid:
                    _save_npz_dict(cache_path, {
                        "actions": np.asarray(sol_actions, dtype=np.int32),
                        "source_algo": np.asarray(algo),
                        "source_kind": np.asarray(source_kind),
                    })

                # Use the JIT'd lax.scan path the random rollouts already
                # take (`_run_eval_rollouts_jax`): pre-roll the C++ env once
                # along the cached solution actions, then scan the model
                # over the (state, action, real_next) trajectory in a
                # single JAX call. ~16x faster than the per-step Python
                # loop (`_run_eval_rollout`) per the comment at the random
                # rollout site above. Single-episode shape `(1, n_steps)`.
                actions_2d_search = np.asarray(sol_actions, dtype=np.int32)[None]
                r_jax = _run_eval_rollouts_jax(
                    model, params, json_str, level_i, n_objs,
                    max_C, max_H, max_W,
                    n_episodes=1,
                    max_steps=len(sol_actions),
                    actions_2d=actions_2d_search,
                    teacher_forced=False,
                    **cond_kwargs,
                )
                # Convert the (n_eps, T) JAX-rollout shape back to the
                # (T,) shape the rest of this code path expects.
                bits = r_jax["wrong_tiles_grid"][0]    # (T,)
                cells = r_jax["wrong_cells_grid"][0]   # (T,)
                first_div = int(r_jax["first_div"][0])
                level_results[algo] = {
                    "error_rate": bits / max(1, r_jax["total_tiles"]),
                    "cell_error_rate": cells / max(1, r_jax["total_cells"]),
                    "wrong_tiles": bits,
                    "wrong_cells": cells,
                    "first_div_step": first_div,
                    "total_tiles": r_jax["total_tiles"],
                    "total_cells": r_jax["total_cells"],
                    "n_steps": len(sol_actions),
                }

            game_results[level_i] = level_results

        results[name] = game_results

        # Print summary for this game
        for level_i, level_results in game_results.items():
            for rtype, metrics in level_results.items():
                wt = metrics.get("mean_wrong_tiles", metrics.get("wrong_tiles"))
                if wt is None:
                    continue
                total = metrics["total_tiles"]
                n = len(wt)
                w_mean = int(round(wt.mean())) if n > 0 else 0
                w_max = int(round(wt.max())) if n > 0 else 0
                fd = metrics.get("mean_first_div", metrics.get("first_div_step"))
                fd_str = f"  first_div={fd:.1f}" if fd is not None else ""
                print(f"  {name} L{level_i} {rtype:<10} "
                      f"wrong: mean={w_mean}  max={w_max}  "
                      f"({n} steps, {total} tiles){fd_str}")

    # Log to wandb
    if wandb.run is not None:
        for name, game_results in results.items():
            for level_i, level_results in game_results.items():
                for rtype, metrics in level_results.items():
                    wt = metrics.get("mean_wrong_tiles", metrics.get("wrong_tiles"))
                    if wt is not None and len(wt) > 0:
                        wandb.log({
                            f"eval/{name}/L{level_i}/{rtype}/mean_wrong": float(wt.mean()),
                            f"eval/{name}/L{level_i}/{rtype}/max_wrong": float(wt.max()),
                        }, commit=False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Flatten to saveable arrays
        save_dict = {}
        for name, game_results in results.items():
            for level_i, level_results in game_results.items():
                for rtype, metrics in level_results.items():
                    key = f"{name}_L{level_i}_{rtype}"
                    er = metrics.get("mean_error_rate", metrics.get("error_rate"))
                    if er is not None:
                        save_dict[f"{key}_error_rate"] = er
                    cer = metrics.get("mean_cell_error_rate",
                                      metrics.get("cell_error_rate"))
                    if cer is not None:
                        save_dict[f"{key}_cell_error_rate"] = cer
                    if "wrong_tiles" in metrics:
                        save_dict[f"{key}_wrong_tiles"] = metrics["wrong_tiles"]
                    if "wrong_cells" in metrics:
                        save_dict[f"{key}_wrong_cells"] = metrics["wrong_cells"]
                    if "mean_wrong_cells" in metrics:
                        save_dict[f"{key}_mean_wrong_cells"] = metrics["mean_wrong_cells"]
                    if "mean_first_div" in metrics:
                        save_dict[f"{key}_mean_first_div"] = np.array(
                            metrics["mean_first_div"])
                    if "first_div_step" in metrics:
                        save_dict[f"{key}_first_div_step"] = np.array(
                            metrics["first_div_step"])
        np.savez(os.path.join(save_dir, "eval_multigame.npz"), **save_dict)
        _write_run_scorecard(save_dir)

    return results


def _write_run_scorecard(save_dir: str) -> None:
    """Render the train/val curve + per-level eval scorecard into ``save_dir``
    so every finished run carries its own figures/tables. Decoupled from the
    training process (subprocess, isolated matplotlib) and never fatal."""
    import subprocess
    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    jobs = [
        ([sys.executable, os.path.join(scripts_dir, "plot_train_val_curves.py"),
          save_dir, "--out", os.path.join(save_dir, "train_val_curves.pdf"),
          "--title", os.path.basename(save_dir.rstrip("/"))],
         "train_val_curves"),
        ([sys.executable, os.path.join(scripts_dir, "summarize_run.py"), save_dir],
         "eval_summary"),
    ]
    for cmd, label in jobs:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            print(f"  wrote {label} into {save_dir}")
        except Exception as e:
            print(f"  [warn] {label} generation failed: {e}")
