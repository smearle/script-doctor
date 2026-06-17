"""Train an NCA world model on a PuzzleScript game.

The NCA learns to predict the next game state given the current state and
player action: f(state_t, action_t) -> state_{t+1}.

Trajectories are collected from random rollouts and search (BFS/A*) via the
C++ PuzzleScript backend.

Usage (run from repo root):
    python nca_wm/train.py --game pipe_bend
    python nca_wm/train.py --games small --conditional --n_hid 128
"""
import argparse
import base64
import io
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so we can import the top-level backends
# (puzzlescript_cpp, puzzlescript_jax) while living under nca_wm/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser
from nca_wm.tokenize_game import (
    tokenize_game, get_game_tree_from_js,
    VOCAB_SIZE_EXT,
)
from nca_wm.scaling_gallery_presets import SCALING_GALLERY_PRESETS
from nca_wm.state_ops import (
    _pack_states, _unpack_states, _pad_offsets, _pad_obs, _pad_packed,
    _dats_to_multihot_batch, _multihot_to_objects,
)
from nca_wm.data_collection import (
    _cache_dir, _load_npz_dict, _save_npz_dict,
    _enabled_action_count, _rollout_history,
    build_predecessor_adjacency, sample_backward_paths,
    _solution_from_sol_dir, _solution_from_transitions_cache,
    collect_unique_transitions, collect_multigame_dataset,
    collect_multigame_dataset_synthetic, _build_sprite_tensor,
)
from nca_wm.inference import (
    _wm_p, make_apply_fn, make_eval_forward, _pad_state_for_model, _unpad_pred,
)
from nca_wm.gif_rendering import (
    _render_training_gif, render_multigame_gifs, play_world_model,
    render_rollout_comparison, render_post_training_gifs,
)

N_ACTIONS = 5

# Preset game sets for multi-game training
MULTI_GAME_PRESETS = {
    "synthetic": [
        "push_sokoban_synthetic",    # standard push
        "swap_sokoban_synthetic",    # player-box swap
        "vanish_sokoban_synthetic",  # box vanishes on contact
    ],
    "small": [
        "nekopuzzle",              # 3 objs,  7x8
        "notsnake",                # 3 objs,  5x8
        "blocks",                  # 4 objs, 11x13
        "sokoban_basic",           # 5 objs,  7x6
        "sokoban_match3",          # 5 objs,  7x9
        "Zen_Puzzle_Garden",       # 6 objs, 12x12
        "Multi-word_Dictionary_Game",  # 7 objs,  7x9
        "kettle",                  # 8 objs, 13x15
        "Travelling_salesman",     # 9 objs,  5x5
    ],
    # Nested subsets of "small" for dataset-scaling experiments. Each larger
    # set is a superset of the previous; games are ordered to include a mix of
    # "well-fit" (sokoban_basic, blocks) and "poorly-fit" (nekopuzzle,
    # Travelling_salesman) from the existing cond_vs_uncond run.
    "scaling_1": ["sokoban_basic"],
    "scaling_1_neko": ["nekopuzzle"],  # sanity-check: can the model fit nekopuzzle alone?
    "scaling_2": ["sokoban_basic", "nekopuzzle"],
    # Singletons of games used in the global-rules architecture experiment.
    # Each tests one rule type in isolation.
    "global_neko": ["nekopuzzle"],          # `...` only
    "global_mazezam": ["MazezaM"],           # `...` only (longer rows)
    "global_constellationz": ["constellationz"],  # `[X][Y]` only
    "global_clearing": ["Clearing_Space"],   # `[X][Y]` only
    "global_nirvana": ["Nirvana"],           # both `...` and `[X][Y]` (separate rules)
    "global_n_step_punt": ["N_Step_Punt"],   # both on the SAME rule line
    "global_sokoban_ctrl": ["sokoban_basic"],  # control: no global rules
    # Bottleneck singletons from scaling_6 multi-game experiment:
    "global_kettle": ["kettle"],
    "global_zen": ["Zen_Puzzle_Garden"],
    "global_travelling_salesman": ["Travelling_salesman"],
    "scaling_4": ["sokoban_basic", "nekopuzzle", "blocks", "Travelling_salesman"],
    "scaling_6": ["sokoban_basic", "nekopuzzle", "blocks", "Travelling_salesman",
                  "Zen_Puzzle_Garden", "kettle"],
    # scaling_14 = "small" (9) + 5 representative mid-sized additions.
    # Binary-search between small (9, no collapse) and scaling_large (19,
    # total collapse). Excludes the biggest additions (constellationz,
    # the_undertaking) and the tiniest ones (sumo=300, wrappingrecipe=125,
    # rigidfail1=3K) so the balanced-sampling pools stay healthy.
    "scaling_14": [
        # original "small" 9:
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        # +5 mid-sized gallery additions:
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "scriptcross", "Modality",
    ],
    # scaling_large: "small" plus simple gallery additions (1-4 rules each),
    # picked for low complexity first so the model has a path to grow.
    "scaling_large": [
        # original "small" 9:
        "nekopuzzle", "notsnake", "blocks", "sokoban_basic", "sokoban_match3",
        "Zen_Puzzle_Garden", "Multi-word_Dictionary_Game", "kettle",
        "Travelling_salesman",
        # simple gallery additions (11 more):
        "blank", "sumo", "the_undertaking", "wrappingrecipe",
        "Collapsable_Sokoban", "Love_and_Pieces", "actiontest",
        "rigidfail1", "scriptcross", "Modality", "constellationz",
    ],
}

# Curated scaling_gallery_v{1..5} lists live in their own dependency-free module
# (see its docstring for provenance; they are superseded by --n_per_rule_games).
# Merge them in so `--games scaling_gallery_vN` resolves exactly as before.
MULTI_GAME_PRESETS.update(SCALING_GALLERY_PRESETS)


# ---------------------------------------------------------------------------
# 2. NCA world model classes — extracted to nca_wm/models.py
# ---------------------------------------------------------------------------
from nca_wm.models import (
    _pool_features,
    NCAWorldModel,
    GameSpecEncoder,
    ConditionalNCAWorldModel,
)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def _weighted_bce_with_logits(logit, label, pos_weight):
    """Numerically stable BCE-with-logits that upweights the positive class.

    loss = pos_weight * label * softplus(-logit) + (1 - label) * softplus(logit)
    """
    return pos_weight * label * jax.nn.softplus(-logit) + (1.0 - label) * jax.nn.softplus(logit)


def make_train_step(model, optimizer, conditional=False,
                    win_loss_weight: float = 1.0, win_pos_weight: float = 1.0,
                    sprite_loss_weight: float = 0.0,
                    change_loss_weight: float = 0.0,
                    decoder=None,
                    token_decoder_loss_weight: float = 0.0,
                    use_vq: bool = False,
                    vq_commitment_weight: float = 0.25,
                    vq_loss_weight: float = 1.0,
                    vq_usage_loss_weight: float = 0.0,
                    adaptive_halt: bool = False,
                    halt_prior_p: float = 0.1,
                    halt_kl_weight: float = 0.01,
                    halt_mode: str = "ponder"):
    """Returns a JIT-compiled train step with a win-prediction head.

    If ``sprite_loss_weight > 0`` (and conditional), also optimizes a sprite-
    decoder MSE loss: sigmoid(sprite_logits) vs target_sprites normalized to
    [0, 1]. ``target_sprites`` is a (B, n_out, 5, 5, 4) uint8 tensor gathered
    from per-game sprite tensors by game_id.
    """

    def _heads_loss(logits, win_logit, sprite_logits, states, next_states, wons,
                     target_sprites=None, spatial_mask=None):
        bce = optax.sigmoid_binary_cross_entropy(logits, next_states)
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        correct = (preds == next_states).astype(jnp.float32)
        mask = spatial_mask.astype(bce.dtype)
        mask_sum = jnp.maximum(mask.sum(), 1.0)
        acc = (correct * mask).sum() / mask_sum
        changed = (states != next_states) & (spatial_mask > 0)
        n_changed = changed.sum()
        changed_correct = ((preds == next_states) & changed).sum()
        change_acc = jnp.where(n_changed > 0, changed_correct / n_changed, 1.0)
        # Optionally upweight BCE on cells that actually changed (anti
        # identity-collapse escape hatch). weight = 1 on unchanged cells,
        # (1 + change_loss_weight) on changed cells. At c_l_w=0, identical
        # to uniform mean. At c_l_w=10, changed cells count 11× as much.
        if change_loss_weight > 0:
            weight = 1.0 + change_loss_weight * changed.astype(bce.dtype)
            weight = weight * mask
            state_loss = (bce * weight).sum() / jnp.maximum(weight.sum(), 1.0)
        else:
            state_loss = (bce * mask).sum() / mask_sum
        # Win-head BCE with pos_weight to counter class imbalance
        wons_f = wons.astype(jnp.float32)
        win_bce = _weighted_bce_with_logits(win_logit, wons_f, win_pos_weight).mean()
        win_preds = (jax.nn.sigmoid(win_logit) > 0.5).astype(jnp.float32)
        win_acc = (win_preds == wons_f).mean()
        n_win = wons_f.sum()
        win_tp = ((win_preds == 1.0) & (wons_f == 1.0)).sum()
        win_recall = jnp.where(n_win > 0, win_tp / n_win, 1.0)
        total = state_loss + win_loss_weight * win_bce

        # Sprite-decoder loss (optional). Sprite logits → sigmoid → [0,1] RGBA;
        # target is uint8 normalized to [0,1]. MSE over the whole per-object
        # kernel tensor (including alpha channel).
        if sprite_loss_weight > 0 and target_sprites is not None:
            sprite_pred = jax.nn.sigmoid(sprite_logits)
            sprite_tgt = target_sprites.astype(jnp.float32) / 255.0
            sprite_mse = ((sprite_pred - sprite_tgt) ** 2).mean()
            total = total + sprite_loss_weight * sprite_mse
        else:
            sprite_mse = jnp.asarray(0.0, dtype=jnp.float32)
        return total, (state_loss, acc, change_acc, win_bce, win_acc, win_recall,
                        sprite_mse)

    def _ponder_loss(per_step_logits, per_step_win, per_step_halt_logits,
                      states, next_states, wons, spatial_mask=None):
        """PonderNet-style adaptive-halting loss.

        per_step_logits: (T, B, n_out, H, W) — logits at each NCA step.
        per_step_win:    (T, B)              — win logits at each step.
        per_step_halt_logits: (T, B)         — halt logits at each step.

        Halt distribution: λ_k = sigmoid(halt_logits_k) for k < T; the last
        step is forced halt (λ_T := 1) so Σ_k p_k = 1 exactly.
            p_k = λ_k · Π_{j<k}(1 - λ_j)

        Loss = Σ_k p_k · L_k + halt_kl_weight · KL(p || Geom(halt_prior_p))
        where L_k is the same _heads_loss-style state+win loss at step k.

        Returns (total_loss, aux) where aux mirrors _heads_loss's aux but
        with metrics computed at the *expected* step (Σ_k p_k · metric_k)
        plus an extra (expected_steps, kl) pair.
        """
        T = per_step_logits.shape[0]
        # Halt probs: λ ∈ (0, 1)^{T, B}, with the last row forced to 1.
        lam = jax.nn.sigmoid(per_step_halt_logits)              # (T, B)
        # Build cumulative survival Π_{j<k}(1 - λ_j) along T.
        # surv[k] = Π_{j<k} (1 - λ_j); surv[0] = 1.
        log_one_minus = jnp.log(jnp.clip(1.0 - lam, 1e-6, 1.0))  # (T, B)
        surv = jnp.exp(jnp.concatenate([
            jnp.zeros((1, log_one_minus.shape[1])),
            jnp.cumsum(log_one_minus, axis=0)[:-1],
        ], axis=0))                                             # (T, B)
        # p_k for k<T uses λ_k * surv_k; p_T uses surv_T (forced halt).
        p = lam * surv                                          # (T, B)
        # Replace the last row with the forced-halt mass.
        last_surv = jnp.exp(jnp.sum(log_one_minus[:-1], axis=0))  # (B,)
        p = p.at[-1].set(last_surv)                             # (T, B), Σ_k p_k = 1

        # Per-step, per-batch-element state loss. Critically we keep the
        # batch axis until *after* multiplying by p, so each batch element's
        # halt distribution can pair with its own per-step loss — which is
        # what enables per-input adaptive halting. (The previous version
        # averaged loss + p over batch independently and then took their
        # outer product; that lost the per-element coupling that PonderNet
        # depends on.)
        next_b = next_states[None]                              # (1, B, n_out, H, W)
        states_b = states[None]
        bce = optax.sigmoid_binary_cross_entropy(per_step_logits, jnp.broadcast_to(next_b, per_step_logits.shape))
        mask_b = spatial_mask[None].astype(bce.dtype)
        mask_b = jnp.broadcast_to(mask_b, per_step_logits.shape)
        mask_sum_per_b = jnp.maximum(mask_b.sum(axis=(2, 3, 4)), 1.0)
        if change_loss_weight > 0:
            changed = (states_b != next_b).astype(bce.dtype)
            changed = jnp.broadcast_to(changed, per_step_logits.shape)
            weight = 1.0 + change_loss_weight * changed
            weight = weight * mask_b
            # (T, B): weighted-mean BCE per (step, batch-element).
            state_loss_per_step_per_b = (bce * weight).sum(axis=(2, 3, 4)) / jnp.maximum(weight.sum(axis=(2, 3, 4)), 1.0)
        else:
            state_loss_per_step_per_b = (bce * mask_b).sum(axis=(2, 3, 4)) / mask_sum_per_b

        # Per-step, per-batch-element win BCE.
        wons_f = wons.astype(jnp.float32)
        # _weighted_bce_with_logits returns (B,) for each step's (B,) logits.
        win_bce_per_step_per_b = jax.vmap(
            lambda wl: _weighted_bce_with_logits(wl, wons_f, win_pos_weight)
        )(per_step_win)                                          # (T, B)

        # Per-batch-element total head loss at each step.
        L_per_step_per_b = state_loss_per_step_per_b + win_loss_weight * win_bce_per_step_per_b  # (T, B)
        # Loss aggregation across the T (per-step) axis. Three modes:
        #   ponder    — PonderNet-style: weight by learned halt distribution p.
        #               L = E_b[Σ_k p_k(b) · L_k(b)]. Each example chooses its
        #               own halt step via the halt head; KL regularizer pulls
        #               p toward a geometric prior. Body gets gradient at
        #               every k, weighted — rewards shortcut predictions.
        #   uniform   — mean over k (treats every step's prediction as equally
        #               important). Body must make readout good at *every*
        #               depth — prerequisite for convergence-based stopping at
        #               inference but actively rewards shortcuts even more.
        #   argmax_st — Straight-through argmax: forward computes L only at
        #               k* = argmax_k p_k(b) (one selected step per batch
        #               element); backward gradient on halt logits flows via
        #               soft p (so halt can still learn). Body sees gradient
        #               only through L_{k*}, so it isn't penalized for being
        #               wrong at unselected k's. Combined with the KL prior
        #               this should let the model learn depth-specialised
        #               predictions per instance without shortcut pressure.
        if halt_mode == "uniform":
            L_rec = L_per_step_per_b.mean()  # mean over (T, B)
        elif halt_mode == "argmax_st":
            # Hard one-hot mask of argmax in forward; soft p in backward.
            k_star = jnp.argmax(p, axis=0)                        # (B,)
            mask_hard = jax.nn.one_hot(k_star, T, axis=0)          # (T, B)
            mask = mask_hard + p - jax.lax.stop_gradient(p)
            L_rec = (mask * L_per_step_per_b).sum(axis=0).mean()
        elif halt_mode == "convergence_st":
            # Convergence-based selection. k*(b) = first k where the
            # discrete prediction has converged (fraction of cells whose
            # binary readout flipped between k-1 and k is below
            # halt_prior_p). If never converges within T, falls back to T.
            # No halt-head gradient is meaningful here (the head is unused
            # for selection); body gradient flows only through L_{k*}.
            B_ax = per_step_logits.shape[1]
            preds = (jax.nn.sigmoid(per_step_logits) > 0.5).astype(jnp.float32)
            # diff[k] = fraction of cells changing between step k and k-1
            # for k=1..T-1.
            pred_diff = (preds[1:] != preds[:-1]).astype(jnp.float32)
            diff = (pred_diff * mask_b[:-1]).sum(axis=(2, 3, 4)) / mask_sum_per_b[:-1]
            converged = diff < halt_prior_p                        # (T-1, B)
            # Stack a sentinel "always converged" row at the end so argmax
            # finds the latest step if no earlier convergence happened.
            converged_full = jnp.concatenate(
                [converged, jnp.ones((1, B_ax), dtype=bool)], axis=0)  # (T, B)
            k_star_idx = jnp.argmax(converged_full.astype(jnp.int32), axis=0)
            k_star = jnp.minimum(k_star_idx + 1, T - 1)             # (B,) in 0..T-1
            mask_hard = jax.nn.one_hot(k_star, T, axis=0)           # (T, B)
            L_rec = (mask_hard * L_per_step_per_b).sum(axis=0).mean()
        else:  # "ponder" (default)
            L_rec = (p * L_per_step_per_b).sum(axis=0).mean()
        # Batch-marginal helpers for the reporting metrics below.
        state_loss_per_step = state_loss_per_step_per_b.mean(axis=1)  # (T,)
        win_bce_k = win_bce_per_step_per_b.mean(axis=1)               # (T,)
        p_marginal = p.mean(axis=1)                                    # (T,)

        # KL(p || Geometric(halt_prior_p)) per batch element, mean.
        # prior_k = (1 - halt_prior_p)^(k-1) * halt_prior_p for k < T;
        # prior_T = (1 - halt_prior_p)^(T-1)  (truncation mass).
        ks = jnp.arange(T)
        log_prior = jnp.where(
            ks < T - 1,
            ks * jnp.log1p(-halt_prior_p) + jnp.log(halt_prior_p),
            (T - 1) * jnp.log1p(-halt_prior_p),
        )                                                        # (T,)
        # KL = Σ_k p_k log(p_k / prior_k), averaged over batch.
        log_p = jnp.log(jnp.clip(p, 1e-8, 1.0))                  # (T, B)
        kl_per_b = (p * (log_p - log_prior[:, None])).sum(axis=0)  # (B,)
        kl = kl_per_b.mean()

        total = L_rec + halt_kl_weight * kl

        # Reporting metrics: use the expected step (E[k]+1, since k is
        # 0-indexed) and the expected per-step state metrics.
        exp_step = (p_marginal * (jnp.arange(T) + 1).astype(jnp.float32)).sum()

        # For aux compatibility with _heads_loss, compute expected-state-loss
        # (= L_rec but state-only), plus expected acc and change_acc using
        # the per-step argmax predictions weighted by p.
        preds = (jax.nn.sigmoid(per_step_logits) > 0.5).astype(jnp.float32)
        next_b_full = jnp.broadcast_to(next_b, per_step_logits.shape)
        states_b_full = jnp.broadcast_to(states_b, per_step_logits.shape)
        correct = (preds == next_b_full).astype(jnp.float32)
        acc_k = ((correct * mask_b).sum(axis=(2, 3, 4)) / mask_sum_per_b).mean(axis=1)
        changed_full = ((states_b_full != next_b_full) & (mask_b > 0)).astype(jnp.float32)
        changed_correct_k = (correct * changed_full).sum(axis=(1, 2, 3, 4))
        n_changed_k = changed_full.sum(axis=(1, 2, 3, 4))
        change_acc_k = jnp.where(n_changed_k > 0, changed_correct_k / n_changed_k, 1.0)
        acc = (p_marginal * acc_k).sum()
        change_acc = (p_marginal * change_acc_k).sum()
        # Win metrics from the expected-step.
        win_preds = (jax.nn.sigmoid(per_step_win) > 0.5).astype(jnp.float32)
        win_acc_k = (win_preds == wons_f[None]).astype(jnp.float32).mean(axis=1)
        win_acc = (p_marginal * win_acc_k).sum()
        n_win = wons_f.sum()
        win_tp_k = ((win_preds == 1.0) & (wons_f[None] == 1.0)).astype(jnp.float32).sum(axis=1)
        win_recall_k = jnp.where(n_win > 0, win_tp_k / n_win, 1.0)
        win_recall = (p_marginal * win_recall_k).sum()

        sprite_mse = jnp.asarray(0.0, dtype=jnp.float32)
        state_loss_exp = (p_marginal * state_loss_per_step).sum()
        win_bce_exp = (p_marginal * win_bce_k).sum()

        return total, (state_loss_exp, acc, change_acc, win_bce_exp,
                       win_acc, win_recall, sprite_mse), (exp_step, kl)

    joint = decoder is not None and token_decoder_loss_weight > 0

    def _vq_utilization(vq_indices):
        counts = jnp.bincount(
            vq_indices.reshape(-1),
            length=getattr(model, "vq_codebook_size", 1),
        )
        return (counts > 0).sum().astype(jnp.float32)

    # All three train_step variants append (vq_cb_loss, vq_commit_loss,
    # vq_utilization) to aux as the last positions, regardless of whether VQ
    # is enabled. When VQ is off they are zeros — keeps the loop unpack shape
    # stable.
    if conditional and joint:
        from nca_wm.token_decoder import shift_right, decoder_loss as _dec_loss

        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states, wons,
                        game_tokens, game_masks, target_sprites=None,
                        target_tokens=None, target_token_masks=None,
                        spatial_mask=None, hist_states=None, hist_actions=None,
                        cond_dropout_mask=None):
            def loss_fn(params):
                wm_p, dec_p = params["wm"], params["dec"]
                if use_vq:
                    logits, win_logit, sprite_logits, all_slots, vq_aux = model.apply(
                        wm_p, states, action_onehots, game_tokens, game_masks,
                        return_slots=True, return_vq_aux=True,
                        hist_states=hist_states, hist_actions=hist_actions,
                        cond_dropout_mask=cond_dropout_mask,
                    )
                    (vq_cb_loss, vq_commit_loss, vq_indices,
                     vq_usage_loss, vq_soft_perplexity) = vq_aux
                    vq_util = _vq_utilization(vq_indices)
                else:
                    logits, win_logit, sprite_logits, all_slots = model.apply(
                        wm_p, states, action_onehots, game_tokens, game_masks,
                        return_slots=True,
                        hist_states=hist_states, hist_actions=hist_actions,
                        cond_dropout_mask=cond_dropout_mask,
                    )
                    z = jnp.asarray(0.0, dtype=jnp.float32)
                    vq_cb_loss, vq_commit_loss, vq_util = z, z, z
                    vq_usage_loss, vq_soft_perplexity = z, z
                heads_total, aux = _heads_loss(
                    logits, win_logit, sprite_logits,
                    states, next_states, wons,
                    target_sprites=target_sprites,
                    spatial_mask=spatial_mask,
                )
                # Decoder forward — teacher-forced (shift target right by one).
                shifted = shift_right(target_tokens)
                dec_logits = decoder.apply(dec_p, shifted, all_slots)
                dec_loss_v, dec_acc_v = _dec_loss(
                    dec_logits, target_tokens, target_token_masks,
                )
                total = heads_total + token_decoder_loss_weight * dec_loss_v
                if use_vq:
                    total = total + vq_loss_weight * (
                        vq_cb_loss + vq_commitment_weight * vq_commit_loss
                    )
                    total = total + vq_usage_loss_weight * vq_usage_loss
                return total, aux + (
                    dec_loss_v, dec_acc_v, vq_cb_loss, vq_commit_loss, vq_util,
                    vq_usage_loss, vq_soft_perplexity,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return (params_new, opt_state_new, loss) + aux
    elif conditional:
        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states, wons,
                        game_tokens, game_masks, target_sprites=None,
                        spatial_mask=None, hist_states=None, hist_actions=None,
                        cond_dropout_mask=None):
            def loss_fn(params):
                if use_vq:
                    logits, win_logit, sprite_logits, vq_aux = model.apply(
                        params, states, action_onehots, game_tokens, game_masks,
                        return_vq_aux=True,
                        hist_states=hist_states, hist_actions=hist_actions,
                        cond_dropout_mask=cond_dropout_mask,
                    )
                    (vq_cb_loss, vq_commit_loss, vq_indices,
                     vq_usage_loss, vq_soft_perplexity) = vq_aux
                    vq_util = _vq_utilization(vq_indices)
                elif adaptive_halt:
                    logits, win_logit, sprite_logits, halt_aux = model.apply(
                        params, states, action_onehots, game_tokens, game_masks,
                        hist_states=hist_states, hist_actions=hist_actions,
                        cond_dropout_mask=cond_dropout_mask,
                    )
                    z = jnp.asarray(0.0, dtype=jnp.float32)
                    vq_cb_loss, vq_commit_loss, vq_util = z, z, z
                    vq_usage_loss, vq_soft_perplexity = z, z
                else:
                    logits, win_logit, sprite_logits = model.apply(
                        params, states, action_onehots, game_tokens, game_masks,
                        hist_states=hist_states, hist_actions=hist_actions,
                        cond_dropout_mask=cond_dropout_mask,
                    )
                    z = jnp.asarray(0.0, dtype=jnp.float32)
                    vq_cb_loss, vq_commit_loss, vq_util = z, z, z
                    vq_usage_loss, vq_soft_perplexity = z, z
                if adaptive_halt:
                    per_step_logits, per_step_win, per_step_halt = halt_aux
                    heads_total, aux, _ = _ponder_loss(
                        per_step_logits, per_step_win, per_step_halt,
                        states, next_states, wons,
                        spatial_mask=spatial_mask,
                    )
                else:
                    heads_total, aux = _heads_loss(
                        logits, win_logit, sprite_logits,
                        states, next_states, wons,
                        target_sprites=target_sprites,
                        spatial_mask=spatial_mask,
                    )
                total = heads_total
                if use_vq:
                    total = total + vq_loss_weight * (
                        vq_cb_loss + vq_commitment_weight * vq_commit_loss
                    )
                    total = total + vq_usage_loss_weight * vq_usage_loss
                # Pad dec losses with zeros so the loop unpack is independent
                # of the joint-decoder branch.
                z = jnp.asarray(0.0, dtype=jnp.float32)
                return total, aux + (
                    z, z, vq_cb_loss, vq_commit_loss, vq_util,
                    vq_usage_loss, vq_soft_perplexity,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return (params_new, opt_state_new, loss) + aux
    else:
        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states, wons,
                        spatial_mask=None, hist_states=None, hist_actions=None):
            def loss_fn(params):
                logits, win_logit, sprite_logits = model.apply(
                    params, states, action_onehots,
                    hist_states=hist_states, hist_actions=hist_actions,
                )
                heads_total, aux = _heads_loss(
                    logits, win_logit, sprite_logits,
                    states, next_states, wons,
                    target_sprites=None,
                    spatial_mask=spatial_mask,
                )
                z = jnp.asarray(0.0, dtype=jnp.float32)
                return heads_total, aux + (z, z, z, z, z, z, z)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return (params_new, opt_state_new, loss) + aux

    return train_step


def _atomic_save_checkpoint(save_dir: str, params, total_steps: int,
                             early_stopped: bool = None, n_updates_requested: int = None):
    """Write params.pkl and train_meta.json atomically via tmp-file + os.replace.

    Safe for a concurrent reader (e.g. a --render_only process) to load without
    catching a half-written file.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "params.pkl")
    tmp_ckpt = ckpt_path + ".tmp"
    with open(tmp_ckpt, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    os.replace(tmp_ckpt, ckpt_path)

    meta_path = os.path.join(save_dir, "train_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta["total_steps"] = total_steps
    if early_stopped is not None:
        meta["early_stopped"] = bool(early_stopped)
    if n_updates_requested is not None:
        meta["n_updates_requested"] = int(n_updates_requested)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)


def train(
    model: NCAWorldModel,
    dataset: dict,
    lr: float = 1e-3,
    n_updates: int = 5000,
    batch_size: int = 64,
    seed: int = 0,
    log_interval: int = 100,
    save_dir: str = "nca_wm/logs",
    init_params=None,
    start_step: int = 0,
    patience: int = 0,
    min_delta: float = 1e-4,
    win_loss_weight: float = 1.0,
    win_pos_weight: float = 1.0,
    ckpt_interval: int = 1000,
    game_names: list[str] | None = None,
    per_game_eval_interval: int = 1000,
    per_game_eval_size: int = 256,
    balanced_sampling: bool = False,
    game_infos: list[dict] | None = None,
    gif_interval: int = 0,
    gif_n_steps: int = 15,
    max_padded_shape: tuple[int, int, int] | None = None,
    ps_parser=None,
    grad_clip: float = 0.0,  # 0 disables; >0 clips by global-norm to this value
    sprite_loss_weight: float = 0.0,
    change_loss_weight: float = 0.0,
    lr_schedule: str = "constant",  # "constant" or "cosine"
    lr_min: float = 1e-6,
    token_decoder_loss_weight: float = 0.0,
    decoder_d_model: int = 128,
    decoder_n_layers: int = 4,
    decoder_n_heads: int = 4,
    use_vq: bool = False,
    vq_commitment_weight: float = 0.25,
    vq_loss_weight: float = 1.0,
    vq_usage_loss_weight: float = 0.0,
    halt_prior_p: float = 0.1,
    halt_kl_weight: float = 0.01,
    halt_mode: str = "ponder",
    val_frac: float = 0.0,
    val_eval_interval: int = 0,
    obj_permute_aug: bool = False,
    history: int = 0,
    cond_mask_prob: float = 0.0,
):
    """Train (or resume training) the NCA world model.

    If init_params is provided, resumes from those weights instead of
    initializing from scratch. start_step offsets the step counter for logging.
    Supports both conditional (ConditionalNCAWorldModel) and unconditional models.

    Early stopping (enabled when patience > 0):
        Monitors smoothed change_acc (window = patience * log_interval steps).
        Stops when no improvement of at least min_delta for ``patience``
        consecutive evaluation windows.
    """
    # Any encoder-based model (FiLM, rule-attention, or non-NCA baseline that
    # reuses the rule-slot encoder) counts as "conditional" for dataset/init-
    # signature purposes.
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    from nca_wm.baselines import (
        CNNWorldModel, UNetWorldModel, ViTWorldModel,
    )
    conditional = "game_tokens" in dataset and isinstance(
        model,
        (ConditionalNCAWorldModel, RuleAttnNCAWorldModel,
         CNNWorldModel, UNetWorldModel, ViTWorldModel),
    )

    # Single-game callers (`collect_unique_transitions` directly into train())
    # hand us a flat dict with packed `states` / `next_states`, plus the
    # original W. Wrap into the per-game layout so the rest of train() has
    # one code path.
    if "per_game_states" not in dataset and "states" in dataset:
        s_flat = dataset["states"]                # packed (N, C, H, ceil(W/8))
        ns_flat = dataset["next_states"]
        a_flat = dataset["actions"]
        w_flat = dataset.get("wons", np.zeros(len(s_flat), dtype=np.uint8))
        # Single-game path always carries the real W in the dict; if missing
        # (shouldn't happen) fall back to inferring an upper bound.
        real_W = int(dataset.get("W", s_flat.shape[-1] * 8))
        C, H = int(s_flat.shape[1]), int(s_flat.shape[2])
        dataset = dict(dataset)  # shallow copy; don't mutate caller's dict
        dataset["per_game_states"] = [s_flat]
        dataset["per_game_next_states"] = [ns_flat]
        dataset["per_game_actions"] = [a_flat]
        dataset["per_game_wons"] = [np.asarray(w_flat, dtype=np.uint8)]
        dataset["game_shapes"] = np.array([[C, H, real_W]], dtype=np.int32)
        dataset["per_game_transition_shapes"] = [
            np.tile(np.array([[C, H, real_W]], dtype=np.int32), (len(s_flat), 1))
        ]
        dataset["per_game_n_transitions"] = np.array([len(s_flat)], dtype=np.int64)
        dataset["max_C"] = C
        dataset["max_H"] = H
        dataset["max_W"] = real_W
        # Conditional not expected for single-game path; guard anyway.
        if "per_game_tokens" not in dataset:
            dataset["per_game_tokens"] = np.zeros((1, 1), dtype=np.int32)
            dataset["per_game_masks"] = np.zeros((1, 1), dtype=np.bool_)

    rng = jax.random.PRNGKey(seed)
    # v7 per-game native storage. No flat (N, C_max, H_max, W_max) arrays.
    per_game_states = dataset["per_game_states"]          # list of (N_g, C_g, H_g, W_g)
    per_game_next_states = dataset["per_game_next_states"]
    per_game_actions_np = dataset["per_game_actions"]     # list of (N_g,) int32
    per_game_wons_np = dataset["per_game_wons"]           # list of (N_g,) uint8
    n_games = len(per_game_states)
    # Global max for batch buffer shape (kept constant for JIT)
    max_C = int(dataset["max_C"])
    max_H = int(dataset["max_H"])
    max_W = int(dataset["max_W"])
    # Per-game native (C, H, W)
    game_CHW = dataset["game_shapes"]   # (n_games, 3) int32
    per_game_n = dataset["per_game_n_transitions"]  # (n_games,) int64
    per_game_transition_shapes = dataset.get("per_game_transition_shapes")
    if per_game_transition_shapes is None:
        per_game_transition_shapes = [
            np.tile(np.asarray(game_CHW[g], dtype=np.int32), (int(per_game_n[g]), 1))
            for g in range(n_games)
        ]
    if conditional:
        tokens_np = dataset["per_game_tokens"]   # (n_games, max_tok_len) int32
        masks_np = dataset["per_game_masks"]     # (n_games, max_tok_len) bool

    # === Per-game train/val split on transitions ===
    # Prefer the dataset's pre-carved val indices (`per_game_val_idx`), which
    # collect_multigame_dataset carves from the FULL explored set per level
    # (representative of the true distribution, not the capped train subsample).
    # Datasets without them (single-game flat dict, synthetic) fall back to a
    # deterministic per-game uniform split by `seed`. Held-out indices are NEVER
    # drawn during training-batch sampling. val_frac=0 ⇒ no holdout.
    val_frac = max(0.0, float(val_frac))
    provided_val = dataset.get("per_game_val_idx")
    use_provided = provided_val is not None
    per_game_val_idx: dict[int, np.ndarray] = {}
    per_game_train_mask: dict[int, np.ndarray] = {}  # bool, len=per_game_n[g]
    val_split_summary = []
    for g in range(n_games):
        n_g = int(per_game_n[g])
        if n_g == 0:
            continue
        if use_provided:
            vs = (np.asarray(provided_val[g], dtype=np.int64)
                  if g < len(provided_val) else np.empty(0, np.int64))
            vs = vs[(vs >= 0) & (vs < n_g)]
            if 0 < len(vs) < n_g:
                per_game_val_idx[g] = vs.astype(np.int32)
                mask = np.ones(n_g, dtype=bool); mask[vs] = False
                per_game_train_mask[g] = mask
                val_split_summary.append((g, n_g, len(vs)))
            else:
                per_game_train_mask[g] = np.ones(n_g, dtype=bool)
            continue
        if val_frac <= 0.0:
            per_game_train_mask[g] = np.ones(n_g, dtype=bool)
            continue
        n_val = int(round(n_g * val_frac))
        # Always leave at least one training transition per game.
        n_val = max(0, min(n_g - 1, n_val))
        if n_val == 0:
            per_game_train_mask[g] = np.ones(n_g, dtype=bool)
            continue
        sub_rng = np.random.RandomState(seed * 7919 + g + 1)
        val_sel = sub_rng.choice(n_g, size=n_val, replace=False)
        per_game_val_idx[g] = val_sel.astype(np.int32)
        mask = np.ones(n_g, dtype=bool); mask[val_sel] = False
        per_game_train_mask[g] = mask
        val_split_summary.append((g, n_g, n_val))
    if val_split_summary:
        n_train_tot = sum(int(per_game_train_mask[g].sum())
                          for g in range(n_games)
                          if int(per_game_n[g]) > 0)
        n_val_tot = sum(len(per_game_val_idx.get(g, []))
                         for g in range(n_games))
        src = "dataset-carved val (full explored set)" if use_provided \
            else f"val_frac={val_frac:.3f}"
        print(f"  [{src}] held out {n_val_tot:,} / "
              f"{n_train_tot + n_val_tot:,} transitions across "
              f"{len(val_split_summary)} games")

    # Pre-sample per-game eval subsets (per-game-LOCAL indices into
    # per_game_states[g]) for periodic diagnostics.
    per_game_eval: dict[int, np.ndarray] | None = None
    per_game_train_pool_sizes: list[int] | None = None
    if game_names is not None and len(game_names) > 1:
        per_game_eval = {}
        per_game_train_pool_sizes = []
        for g in range(n_games):
            n_g = int(per_game_n[g])
            if n_g == 0:
                per_game_train_pool_sizes.append(0)
                continue
            sub_rng = np.random.RandomState(seed + 1000 + g)
            # Only sample from TRAIN indices for the per-game train-eval
            # printout, so its accuracy reflects fit on training data.
            train_local = np.nonzero(per_game_train_mask[g])[0].astype(np.int32)
            if len(train_local) == 0:
                per_game_train_pool_sizes.append(0)
                continue
            size = min(len(train_local), per_game_eval_size)
            per_game_eval[g] = sub_rng.choice(train_local, size=size, replace=False)
            per_game_train_pool_sizes.append(len(train_local))
    # The old global-balanced sampler tracked per-game per-batch sizes here.
    # The new size-bucketed sampler computes its own per-bucket sizes below
    # (see `bucket_sizes_per_game`), so this block only handles the no-data
    # edge case.
    if balanced_sampling and sum(1 for g in range(n_games) if int(per_game_n[g]) > 0) < 2:
        print("  [balanced_sampling] Disabled: need >=2 games with data. Using uniform.")
        balanced_sampling = False

    n_data = int(per_game_n.sum())
    mode_str = "conditional" if conditional else "unconditional"
    n_wins = int(sum(int(w.sum()) for w in per_game_wons_np))
    print(f"Training ({mode_str}) on {n_data:,} transitions "
          f"(per-game native shapes, batch-padded to max ({max_C},{max_H},{max_W})), "
          f"batch_size={batch_size}, lr={lr}, wins={n_wins:,}/{n_data:,} "
          f"({100*n_wins/max(1,n_data):.3f}%)")
    print(f"  win_loss_weight={win_loss_weight}, win_pos_weight={win_pos_weight}")
    print("  [padding mask] Loss/metrics always ignore batch-padding outside each sample's real (C,H,W).")

    # === Size-bucketed batching ===
    # Quantize each transition's real (H, W) up to the nearest power-of-2
    # anchor. Each batch is padded only to its sampled bucket's anchor shape,
    # and loss masks are built from the sampled transition's true (C,H,W).
    # Quantizing (vs exact-shape buckets) caps the number of distinct batch
    # shapes JAX must JIT-compile train_step for — the difference between
    # "warmup takes minutes" and "warmup takes hours" on game sets with many
    # distinct shapes (gallery has 80+ unique (H, W) pairs).
    #
    # Power-of-2 anchors (min 8) cover all gallery games in 4-8 active buckets.
    # max_C stays global because the model's n_out is fixed at construction
    # time and the C-dim padding is "predict 0" — easy auxiliary task.
    #
    # Balanced sampling preserves uniform per-game exposure in expectation:
    # bucket probability is the mean, over games, of that game's transition
    # fraction in the bucket; games within the bucket are sampled proportional
    # to the same per-game fractions. Uniform mode samples transitions
    # uniformly across the whole dataset.
    def _next_pow2(x: int, min_val: int = 8) -> int:
        v = max(min_val, int(x))
        # round up to nearest power of 2
        p = 1
        while p < v:
            p <<= 1
        return p

    _bucket_to_game_indices: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    _val_bucket_to_game_indices: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for g in range(n_games):
        if int(per_game_n[g]) == 0:
            continue
        shapes_g = np.asarray(per_game_transition_shapes[g], dtype=np.int32)
        if len(shapes_g) != int(per_game_n[g]):
            raise ValueError(
                f"per_game_transition_shapes[{g}] length {len(shapes_g)} "
                f"!= per_game_n {int(per_game_n[g])}"
            )
        keys_g = np.array([(_next_pow2(h), _next_pow2(w))
                           for _c, h, w in shapes_g], dtype=np.int32)
        train_mask = per_game_train_mask.get(g)
        for key_arr in np.unique(keys_g, axis=0):
            key = (int(key_arr[0]), int(key_arr[1]))
            in_bucket = (keys_g[:, 0] == key[0]) & (keys_g[:, 1] == key[1])
            # Train-only bucket pool (always built; same as before when
            # val_frac=0 since train_mask is all True).
            idx_train = np.nonzero(in_bucket & train_mask)[0]
            if len(idx_train) > 0:
                _bucket_to_game_indices.setdefault(key, {})[g] = idx_train.astype(np.int32)
            # Test bucket pool (only when held-out transitions exist).
            if val_frac > 0.0:
                idx_test = np.nonzero(in_bucket & (~train_mask))[0]
                if len(idx_test) > 0:
                    _val_bucket_to_game_indices.setdefault(key, {})[g] = idx_test.astype(np.int32)

    bucket_hw = sorted(_bucket_to_game_indices.keys())
    bucket_game_indices = [_bucket_to_game_indices[k] for k in bucket_hw]
    bucket_n_games_arr = np.array([len(d) for d in bucket_game_indices], dtype=np.float64)
    bucket_n_trans_arr = np.array(
        [sum(len(idx) for idx in d.values()) for d in bucket_game_indices],
        dtype=np.float64,
    )
    if balanced_sampling:
        n_nonempty_games = max(1, sum(1 for n in per_game_n if int(n) > 0))
        bucket_weights = []
        bucket_game_probs = []
        for d in bucket_game_indices:
            weights = np.array(
                [len(idx) / max(1, int(per_game_n[g])) for g, idx in d.items()],
                dtype=np.float64,
            )
            bucket_weights.append(weights.sum() / n_nonempty_games)
            bucket_game_probs.append(weights / weights.sum())
        _bucket_probs = np.array(bucket_weights, dtype=np.float64)
        _bucket_probs = _bucket_probs / _bucket_probs.sum()
    else:
        bucket_game_probs = []
        _bucket_probs = bucket_n_trans_arr / bucket_n_trans_arr.sum()
        for d in bucket_game_indices:
            weights = np.array([len(idx) for idx in d.values()], dtype=np.float64)
            bucket_game_probs.append(weights / weights.sum())

    # Per-bucket batch_size: forward+backward activation memory grows
    # with batch_size * H_b * W_b. A single (64x64) bucket at batch_size=32
    # blows up to >20 GiB and OOMs even on a 24 GiB card. Cap each bucket
    # at the same `cells-per-batch` budget as the (16x16) reference shape
    # (batch_size_default * 16 * 16 = 8192 by default), then floor at
    # MIN_PER_BUCKET so very-small buckets don't get a meaningless 1-2
    # element batch. JAX already compiles per-(H_b, W_b) shape, so adding
    # per-bucket B_b doesn't multiply compile count — it just changes the
    # already-distinct signature for each bucket.
    MIN_PER_BUCKET = 4
    target_cells = batch_size * 16 * 16
    bucket_batch_sizes = [
        max(MIN_PER_BUCKET, min(batch_size, target_cells // max(1, H_b * W_b)))
        for (H_b, W_b) in bucket_hw
    ]

    # Pre-allocate per-bucket scratch buffers (sized to per-bucket batch_size).
    bucket_buffers = []
    bucket_a_bufs = []
    bucket_w_bufs = []
    bucket_g_bufs = []
    for (H_b, W_b), bs_b in zip(bucket_hw, bucket_batch_sizes):
        buf = {
            "s":  np.zeros((bs_b, max_C, H_b, W_b), dtype=np.uint8),
            "ns": np.zeros((bs_b, max_C, H_b, W_b), dtype=np.uint8),
            "mask": np.zeros((bs_b, max_C, H_b, W_b), dtype=np.uint8),
        }
        bucket_buffers.append(buf)
        bucket_a_bufs.append(np.zeros((bs_b,), dtype=np.int32))
        bucket_w_bufs.append(np.zeros((bs_b,), dtype=np.uint8))
        bucket_g_bufs.append(np.zeros((bs_b,), dtype=np.int32))

    bucket_summary = ", ".join(
        f"({h}x{w}):{int(k)}g/{int(n):,}t/B={bs}"
        for (h, w), k, n, bs in zip(bucket_hw, bucket_n_games_arr,
                                     bucket_n_trans_arr, bucket_batch_sizes)
    )
    print(f"  [size_buckets] {len(bucket_hw)} (H,W) bucket(s) [B=per-bucket "
          f"batch_size, capped at activation budget {target_cells} cells/batch]: "
          f"{bucket_summary}")

    bucket_games_lists = [list(d.keys()) for d in bucket_game_indices]

    # Init or resume model
    rng, init_rng = jax.random.split(rng)
    dummy_state = jnp.zeros((1, max_C, max_H, max_W), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, N_ACTIONS), dtype=jnp.float32)
    # When history is on, init must include the history channels so the embed
    # layer is sized for the full (current + k·history) input width.
    dummy_hist_kw = {}
    if int(history) > 0:
        dummy_hist_kw = dict(
            hist_states=jnp.zeros((1, int(history), max_C, max_H, max_W),
                                  dtype=jnp.float32),
            hist_actions=jnp.zeros((1, int(history)), dtype=jnp.int32),
        )

    # Decoder is wired in only for rule_attn + token_decoder_loss_weight > 0.
    joint_decoder = None
    is_rule_attn = isinstance(model, RuleAttnNCAWorldModel)
    if conditional and is_rule_attn and token_decoder_loss_weight > 0:
        from nca_wm.token_decoder import SlotTokenDecoder
        max_tok_len = tokens_np.shape[1]
        joint_decoder = SlotTokenDecoder(
            vocab_size=model.vocab_size,
            max_seq_len=max_tok_len,
            d_model=decoder_d_model,
            n_heads=decoder_n_heads,
            n_layers=decoder_n_layers,
            d_slot=model.d_slot,
        )

    if init_params is not None:
        params = init_params
        print(f"Resuming from step {start_step:,}")
    else:
        if conditional:
            max_tok_len = tokens_np.shape[1]
            dummy_tokens = jnp.zeros((1, max_tok_len), dtype=jnp.int32)
            dummy_mask = jnp.zeros((1, max_tok_len), dtype=jnp.bool_)
            wm_params = model.init(init_rng, dummy_state, dummy_action,
                                   dummy_tokens, dummy_mask, **dummy_hist_kw)
            if joint_decoder is not None:
                rng, dec_init_rng = jax.random.split(rng)
                dummy_slots = jnp.zeros((1, model.n_slots, model.d_slot),
                                        dtype=jnp.float32)
                dec_params = joint_decoder.init(dec_init_rng, dummy_tokens, dummy_slots)
                params = {"wm": wm_params, "dec": dec_params}
            else:
                params = wm_params
        else:
            params = model.init(init_rng, dummy_state, dummy_action,
                                 **dummy_hist_kw)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model params: {n_params:,}")
    if joint_decoder is not None:
        n_wm = sum(p.size for p in jax.tree.leaves(_wm_p(params)))
        n_dec = n_params - n_wm
        print(f"  [joint decoder] wm={n_wm:,}  decoder={n_dec:,}  "
              f"(token_decoder_loss_weight={token_decoder_loss_weight})")

    # Optional LR schedule: constant by default (lr stays at the --lr value
    # the whole run). With --lr_schedule cosine, anneals lr from --lr down
    # to --lr_min over n_updates steps. Useful for sharp-minimum cases
    # where the model walks out of optima without decay.
    if lr_schedule == "cosine":
        lr_fn = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=n_updates, alpha=lr_min / max(lr, 1e-12)
        )
    else:
        lr_fn = lr

    if grad_clip > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(lr_fn),
        )
    else:
        optimizer = optax.adam(lr_fn)
    opt_state = optimizer.init(params)
    if change_loss_weight > 0:
        print(f"  [change_loss_weight={change_loss_weight}] changed-cell BCE upweighted "
              f"— each changed cell counts {1+change_loss_weight:.1f}× vs unchanged.")
    train_step = make_train_step(
        model, optimizer, conditional=conditional,
        win_loss_weight=win_loss_weight, win_pos_weight=win_pos_weight,
        sprite_loss_weight=sprite_loss_weight,
        change_loss_weight=change_loss_weight,
        decoder=joint_decoder,
        token_decoder_loss_weight=token_decoder_loss_weight,
        use_vq=use_vq,
        vq_commitment_weight=vq_commitment_weight,
        vq_loss_weight=vq_loss_weight,
        vq_usage_loss_weight=vq_usage_loss_weight,
        adaptive_halt=getattr(model, "adaptive_halt", False),
        halt_prior_p=halt_prior_p,
        halt_kl_weight=halt_kl_weight,
        halt_mode=halt_mode,
    )

    # Object-permutation augmentation: precompute the CH-token id table once.
    # CH<i> token ids occupy two ranges in the V2 vocab (CH0..63 in BASE,
    # CH64..MAX_CHANNELS_V2-1 in EXT extension). Building a lookup once means
    # the per-batch remap is a single numpy fancy-indexing operation.
    obj_permute_ch_ids = None
    if obj_permute_aug and conditional:
        from nca_wm.tokenize_game import VOCAB, MAX_CHANNELS_V2 as _MC
        obj_permute_ch_ids = np.array(
            [VOCAB[f"CH{i}"] for i in range(_MC)], dtype=np.int32)
        print(f"  [obj_permute_aug] enabled: per-batch random permutation of "
              f"{_MC} object channels (CH-tokens + state/mask axis-1)")
    per_game_sprites_np = dataset.get("per_game_sprites")
    eval_forward = make_eval_forward(model, conditional=conditional)
    apply_fn = make_apply_fn(model)  # reused by intermittent gif rendering

    os.makedirs(save_dir, exist_ok=True)
    # Note: RUNNING.pid lock is written earlier in main() before dataset load,
    # so parallel launchers see the lock before they decide to launch.

    losses, accs, change_accs = [], [], []
    state_losses, win_losses, win_accs, win_recalls = [], [], [], []
    sprite_mses: list[float] = []
    dec_losses: list[float] = []
    dec_accs: list[float] = []
    vq_cb_losses: list[float] = []
    vq_commit_losses: list[float] = []
    vq_utils: list[float] = []
    vq_usage_losses: list[float] = []
    vq_soft_perplexities: list[float] = []
    # per_game_log[game_id] = list of dicts (step, loss, acc, change_acc)
    per_game_log: dict[int, list[dict]] = {g: [] for g in (per_game_eval or {})}
    # Held-out per-transition test set: list of {step, loss, acc, change_acc}
    # aggregated across all test transitions; written to curves npz alongside
    # the train losses. Empty when val_frac == 0.
    val_log: list[dict] = []
    if val_eval_interval <= 0:
        # Default cadence: tie to log_interval so train/val curves share x.
        val_eval_interval = log_interval
    # Cap how many test transitions we evaluate per cadence step. Iterating all
    # test transitions every step would dominate wallclock once the held-out
    # set grows. K=512 per game stays cheap and still gives a stable estimate.
    val_eval_max_per_game = 512
    t0 = time.time()
    np_rng = np.random.RandomState(seed + start_step)

    # --- History (Option A) -------------------------------------------------
    # Per-game predecessor adjacency for backward path sampling. Built once on
    # each game's packed transition set; entries are predecessor row lists (a
    # DAG — non-injective dynamics give multiple predecessors per state). At
    # batch time we sample a fresh length-`hist_k` backward path per target,
    # which decorrelates history from the current state (augmentation).
    hist_k = int(history)
    pred_lists_g = None
    _hist_stats = {"miss": 0, "total": 0}
    if hist_k > 0:
        pred_lists_g = []
        n_roots = 0
        n_rows = 0
        for g in range(n_games):
            pl = build_predecessor_adjacency(
                per_game_states[g], per_game_next_states[g])
            pred_lists_g.append(pl)
            n_roots += sum(1 for p in pl if len(p) == 0)
            n_rows += len(pl)
        if n_rows:
            print(f"  history={hist_k}: predecessor adjacency over {n_rows:,} "
                  f"transitions; {n_roots:,} ({100*n_roots/n_rows:.1f}%) have no "
                  f"in-cache predecessor (episode-starts + write-cap holes).")

    def _gather_history(target_pairs, C_buf, H_buf, W_buf, rng):
        """Sample length-`hist_k` backward trajectories for the given target
        (game_id, local_row) pairs and pack the predecessor (state, action)
        pairs into bucket-shaped buffers, oldest→newest (col hist_k-1 is the
        immediate predecessor). Returns (hist_s, hist_a), or (None, None) when
        history is off. Missing steps stay all-zero (masked by the model)."""
        if hist_k <= 0:
            return None, None
        nb = len(target_pairs)
        hist_s = np.zeros((nb, hist_k, C_buf, H_buf, W_buf), dtype=np.uint8)
        hist_a = np.zeros((nb, hist_k), dtype=np.int32)
        for i, (g, j) in enumerate(target_pairs):
            rows, miss = sample_backward_paths(
                pred_lists_g[g], [int(j)], hist_k, rng)
            rows = rows[0]
            _hist_stats["miss"] += int(miss[0])
            _hist_stats["total"] += 1
            W_store = int(game_CHW[g, 2])
            for t in range(hist_k):
                r = int(rows[t])
                if r < 0:
                    continue
                C_r, H_r, W_r = (int(x) for x in
                                 per_game_transition_shapes[g][r])
                s_full = _unpack_states(per_game_states[g][r], W_store)
                oy, _ = _pad_offsets(H_r, s_full.shape[1])
                ox, _ = _pad_offsets(W_r, W_store)
                hist_s[i, t, :C_r, :H_r, :W_r] = \
                    s_full[:C_r, oy:oy + H_r, ox:ox + W_r]
                hist_a[i, t] = per_game_actions_np[g][r]
        return hist_s, hist_a

    gif_enabled = (
        gif_interval > 0 and game_infos is not None
        and max_padded_shape is not None
        and ps_parser is not None
    )

    # Pre-compile renderer once (heavy: loads JS engine + sprite data).
    _gif_backend = None
    if gif_enabled:
        try:
            _gif_backend = CppPuzzleScriptBackend()
            _gif_backend.compile_game(ps_parser, game_infos[0]["name"])
        except Exception as e:
            print(f"  [gif] backend compile failed: {type(e).__name__}: {e}")
            gif_enabled = False

    def _maybe_render_gif(global_step: int, avg_cerr: float | None = None):
        if not gif_enabled:
            return
        g = game_infos[0]
        gt, gm = None, None
        if conditional:
            tlen = tokens_np.shape[1]
            tids = g.get("token_ids", [])
            padded = np.zeros(tlen, dtype=np.int32)
            mask = np.zeros(tlen, dtype=np.bool_)
            padded[:len(tids)] = tids
            mask[:len(tids)] = True
            gt, gm = padded, mask
        mC, mH, mW = max_padded_shape
        # Step is in filename; banner only needs the rollout-step counter
        # (added inside _render_training_gif).
        banner = ""
        path = os.path.join(save_dir, "train_gifs",
                             f"{g['name']}_step{global_step:08d}.gif")
        try:
            _render_training_gif(
                apply_fn, _wm_p(params), g,
                max_C=mC, max_H=mH, max_W=mW,
                save_path=path, n_steps=gif_n_steps,
                seed=global_step,
                conditional=conditional,
                game_tokens=gt, game_mask=gm,
                banner_text=banner,
                backend_render=_gif_backend,
            )
        except Exception as e:
            print(f"  [gif] render failed: {type(e).__name__}: {e}")

    # Early stopping state (monitors smoothed loss — less noisy than change_acc)
    best_loss = float("inf")
    patience_ref_loss = float("inf")  # reset when avg_loss improves by ≥ min_delta
    patience_counter = 0
    early_stopped = False

    # Render a GIF at step 0 (untrained baseline) if enabled.
    if gif_enabled and start_step == 0:
        _maybe_render_gif(start_step)

    def _sample_bucket_batch():
        """Pick a bucket, sample its per-bucket batch_size rows from games
        in that bucket, and populate the bucket's preallocated buffers.
        Returns:
            (b_idx, batch_s, batch_ns, batch_mask, batch_a, batch_w, batch_g)
        batch_s, batch_ns have shape (B_b, max_C, H_b, W_b) where B_b
        depends on the bucket (smaller for large H,W to keep activation
        memory bounded)."""
        b_idx = int(np_rng.choice(len(bucket_hw), p=_bucket_probs))
        bs_b = bucket_batch_sizes[b_idx]
        games_b = bucket_games_lists[b_idx]
        game_indices_b = bucket_game_indices[b_idx]
        game_probs_b = bucket_game_probs[b_idx]
        s_buf = bucket_buffers[b_idx]["s"]
        ns_buf = bucket_buffers[b_idx]["ns"]
        mask_buf = bucket_buffers[b_idx]["mask"]
        a_buf = bucket_a_bufs[b_idx]
        w_buf = bucket_w_bufs[b_idx]
        g_buf = bucket_g_bufs[b_idx]
        s_buf.fill(0); ns_buf.fill(0)
        mask_buf.fill(0)

        # Build game_ids and local_idx for this true-size bucket. In balanced
        # mode, `game_probs_b` is proportional to each game's within-game
        # transition fraction for this bucket; in uniform mode it is
        # proportional to raw transition count in this bucket.
        game_choice = np_rng.choice(len(games_b), size=bs_b, p=game_probs_b)
        local_idx = np.empty(bs_b, dtype=np.int32)
        for k, g in enumerate(games_b):
            rows = np.nonzero(game_choice == k)[0]
            if len(rows) == 0:
                continue
            idx_pool = game_indices_b[g]
            local_idx[rows] = idx_pool[np_rng.randint(0, len(idx_pool), size=len(rows))]
        game_ids = np.array([games_b[k] for k in game_choice], dtype=np.int32)

        # Fill bucket-shaped buffers. per_game_states[g] is bitpacked along W
        # — unpack just the sampled row and write into the (H_g, W_g) corner
        # of the bucket buffer; surrounding cells stay zero from `s_buf.fill(0)`
        # above. Per-row unpack allocates ~C_g·H_g·W_g uint8s; the merged
        # in-RAM dataset stays packed.
        for i in range(bs_b):
            g = int(game_ids[i]); j = int(local_idx[i])
            W_store = int(game_CHW[g, 2])
            C_real, H_real, W_real = (int(x) for x in per_game_transition_shapes[g][j])
            s_full = _unpack_states(per_game_states[g][j], W_store)
            ns_full = _unpack_states(per_game_next_states[g][j], W_store)
            oy, _ = _pad_offsets(H_real, s_full.shape[1])
            ox, _ = _pad_offsets(W_real, W_store)
            s_buf[i, :C_real, :H_real, :W_real] = s_full[:C_real, oy:oy + H_real, ox:ox + W_real]
            ns_buf[i, :C_real, :H_real, :W_real] = ns_full[:C_real, oy:oy + H_real, ox:ox + W_real]
            mask_buf[i, :C_real, :H_real, :W_real] = 1
            a_buf[i] = per_game_actions_np[g][j]
            w_buf[i] = per_game_wons_np[g][j]
            g_buf[i] = g
        # History (Option A): sample a backward trajectory per target and pack
        # predecessor (state, action) pairs into bucket-shaped buffers.
        hist_s = hist_a = None
        if hist_k > 0:
            pairs = [(int(game_ids[i]), int(local_idx[i])) for i in range(bs_b)]
            hist_s, hist_a = _gather_history(
                pairs, s_buf.shape[1], s_buf.shape[2], s_buf.shape[3], np_rng)
        return b_idx, s_buf, ns_buf, mask_buf, a_buf, w_buf, g_buf, hist_s, hist_a

    for step in range(n_updates):
        b_idx, _bs, _bns, _bm, _ba, _bw, _bg, _bhs, _bha = _sample_bucket_batch()
        game_ids_batch = _bg

        # Object-permutation augmentation: sample one π per batch, apply to
        # state/next-state/mask channel axis. Token remap is applied below
        # after gathering tokens_np[game_ids_batch].
        _gt_remap = None
        if obj_permute_ch_ids is not None:
            buf_C = _bs.shape[1]
            perm_full = np_rng.permutation(obj_permute_ch_ids.shape[0])
            # Restrict permutation to the subset that actually exists in the
            # state buffer (channels [0..buf_C-1]); leave higher channels alone.
            perm_buf = perm_full[:buf_C]
            # Sanity: ensure perm_buf only references in-buffer channels.
            in_buf = perm_buf < buf_C
            # Out-of-buffer slots get identity (i.e. swap with self) so we
            # don't try to gather from a nonexistent channel.
            perm_buf = np.where(in_buf, perm_buf, np.arange(buf_C))
            _bs = _bs[:, perm_buf]
            _bns = _bns[:, perm_buf]
            if _bm is not None:
                _bm = _bm[:, perm_buf]
            # History states share the channel axis (axis 2: B,k,C,H,W) — apply
            # the same object permutation so they stay consistent with `s`.
            if _bhs is not None:
                _bhs = _bhs[:, :, perm_buf]
            # Build full vocab-size remap table for token rewrite below.
            _gt_remap = np.arange(int(obj_permute_ch_ids.max()) + 1, dtype=np.int32)
            # Allocate at least vocab-max + 1 by re-allocating against actual gt range
            # — easier: build a dict-style remap and apply via np.where chain.
            # Cleaner: use ch_token_ids as both src/dst to make a sparse remap.
            _gt_remap = obj_permute_ch_ids  # used by remap closure below.

        s = jnp.array(_bs, dtype=jnp.float32)
        a_int = _ba
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a_int])
        ns = jnp.array(_bns, dtype=jnp.float32)
        w = jnp.array(_bw, dtype=jnp.float32)
        spatial_mask = jnp.array(_bm, dtype=jnp.float32)
        hist_states_j = (jnp.array(_bhs, dtype=jnp.float32)
                         if _bhs is not None else None)
        hist_actions_j = (jnp.array(_bha, dtype=jnp.int32)
                          if _bha is not None else None)

        if conditional:
            _gt_np = tokens_np[game_ids_batch]
            if obj_permute_ch_ids is not None:
                # Apply CH-token remap: build a sparse lookup table over the
                # vocab, then use np.take. Vocab-size is small (~640) and
                # the lookup happens once per batch, so this is cheap.
                vocab_size = int(_gt_np.max()) + 1
                vocab_size = max(vocab_size, int(obj_permute_ch_ids.max()) + 1)
                lut = np.arange(vocab_size, dtype=_gt_np.dtype)
                src_ids = obj_permute_ch_ids
                dst_ids = obj_permute_ch_ids[perm_full]
                lut[src_ids] = dst_ids
                _gt_np = lut[_gt_np]
            gt = jnp.array(_gt_np)
            gm = jnp.array(masks_np[game_ids_batch])
            # Classifier-free-guidance-style rule dropout. When cond_mask_prob
            # > 0, draw a per-example Bernoulli mask (True = withhold the rule
            # encoding this step) so the same weights learn both conditional
            # and marginal dynamics. prob == 0 → None → unmasked forward pass
            # (bit-identical to before this feature).
            cond_dropout_mask = None
            if cond_mask_prob > 0.0:
                cond_dropout_mask = jnp.array(
                    np_rng.random(s.shape[0]) < cond_mask_prob)
            # Gather per-batch target sprites by game_id when sprite loss is on.
            target_sprites = None
            if sprite_loss_weight > 0 and per_game_sprites_np is not None:
                target_sprites = jnp.array(per_game_sprites_np[game_ids_batch])
            if joint_decoder is not None:
                # Recon target = the same token sequence the encoder consumed.
                tgt_tokens = gt
                tgt_mask = gm
                (params, opt_state, loss, state_loss, acc, change_acc,
                 win_loss, win_acc, win_recall, sprite_mse,
                 dec_loss, dec_acc,
                 vq_cb_loss, vq_commit_loss, vq_util,
                 vq_usage_loss, vq_soft_perplexity) = train_step(
                    params, opt_state, s, a_oh, ns, w, gt, gm,
                    target_sprites, tgt_tokens, tgt_mask, spatial_mask,
                    hist_states=hist_states_j, hist_actions=hist_actions_j,
                    cond_dropout_mask=cond_dropout_mask,
                )
            else:
                (params, opt_state, loss, state_loss, acc, change_acc,
                 win_loss, win_acc, win_recall, sprite_mse,
                 dec_loss, dec_acc,
                 vq_cb_loss, vq_commit_loss, vq_util,
                 vq_usage_loss, vq_soft_perplexity) = train_step(
                    params, opt_state, s, a_oh, ns, w, gt, gm, target_sprites,
                    spatial_mask,
                    hist_states=hist_states_j, hist_actions=hist_actions_j,
                    cond_dropout_mask=cond_dropout_mask,
                )
        else:
            (params, opt_state, loss, state_loss, acc, change_acc,
             win_loss, win_acc, win_recall, sprite_mse,
             dec_loss, dec_acc,
             vq_cb_loss, vq_commit_loss, vq_util,
             vq_usage_loss, vq_soft_perplexity) = train_step(
                params, opt_state, s, a_oh, ns, w, spatial_mask,
                hist_states=hist_states_j, hist_actions=hist_actions_j,
            )
        losses.append(float(loss))
        accs.append(float(acc))
        change_accs.append(float(change_acc))
        state_losses.append(float(state_loss))
        win_losses.append(float(win_loss))
        win_accs.append(float(win_acc))
        win_recalls.append(float(win_recall))
        sprite_mses.append(float(sprite_mse))
        dec_losses.append(float(dec_loss))
        dec_accs.append(float(dec_acc))
        vq_cb_losses.append(float(vq_cb_loss))
        vq_commit_losses.append(float(vq_commit_loss))
        vq_utils.append(float(vq_util))
        vq_usage_losses.append(float(vq_usage_loss))
        vq_soft_perplexities.append(float(vq_soft_perplexity))

        global_step = start_step + step + 1
        # Atomic periodic checkpoint so a concurrent --render_only process can load
        if ckpt_interval > 0 and global_step % ckpt_interval == 0:
            _atomic_save_checkpoint(save_dir, params, global_step)
        if global_step % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            avg_state_loss = np.mean(state_losses[-log_interval:])
            avg_err = 1.0 - np.mean(accs[-log_interval:])
            avg_cerr = 1.0 - np.mean(change_accs[-log_interval:])
            avg_win_loss = np.mean(win_losses[-log_interval:])
            avg_win_err = 1.0 - np.mean(win_accs[-log_interval:])
            avg_win_recall = np.mean(win_recalls[-log_interval:])
            avg_sprite_mse = np.mean(sprite_mses[-log_interval:]) if sprite_mses else 0.0
            avg_dec_loss = np.mean(dec_losses[-log_interval:]) if dec_losses else 0.0
            avg_dec_acc = np.mean(dec_accs[-log_interval:]) if dec_accs else 0.0
            avg_vq_cb = np.mean(vq_cb_losses[-log_interval:]) if vq_cb_losses else 0.0
            avg_vq_commit = np.mean(vq_commit_losses[-log_interval:]) if vq_commit_losses else 0.0
            avg_vq_util = np.mean(vq_utils[-log_interval:]) if vq_utils else 0.0
            avg_vq_usage = np.mean(vq_usage_losses[-log_interval:]) if vq_usage_losses else 0.0
            avg_vq_perp = np.mean(vq_soft_perplexities[-log_interval:]) if vq_soft_perplexities else 0.0
            elapsed = time.time() - t0
            sprite_bit = f"  sprite_mse={avg_sprite_mse:.4e}" if sprite_loss_weight > 0 else ""
            dec_bit = (f"  dec_loss={avg_dec_loss:.4e}  dec_acc={avg_dec_acc:.3f}"
                       if joint_decoder is not None else "")
            vq_bit = (f"  vq_cb={avg_vq_cb:.4e}  vq_commit={avg_vq_commit:.4e}"
                      f"  vq_util={avg_vq_util:.1f}  vq_usage={avg_vq_usage:.4e}"
                      f"  vq_perp={avg_vq_perp:.1f}"
                      if use_vq else "")
            hist_bit = ""
            if hist_k > 0 and _hist_stats["total"] > 0:
                # Fraction of sampled history steps that hit a dead end before
                # k (episode-starts + write-cap holes). High values flag holey
                # trajectories (Phase 2 ancestor-closed capping would fix them).
                hole_frac = _hist_stats["miss"] / _hist_stats["total"]
                hist_bit = f"  hist_hole={hole_frac:.3f}"
            print(f"  step {global_step:,}/{start_step + n_updates:,}  loss={avg_loss:.4e}  "
                  f"state_loss={avg_state_loss:.4e}  err={avg_err:.4e}  "
                  f"change_err={avg_cerr:.4e}  win_loss={avg_win_loss:.4e}  "
                  f"win_err={avg_win_err:.4e}  win_recall={avg_win_recall:.3f}"
                  f"{sprite_bit}{dec_bit}{vq_bit}{hist_bit}  ({elapsed:.1f}s)")
            if wandb.run is not None:
                wandb_log = {
                    "train/loss": avg_loss,
                    "train/state_loss": avg_state_loss,
                    "train/err": avg_err,
                    "train/change_err": avg_cerr,
                    "train/win_loss": avg_win_loss,
                    "train/win_err": avg_win_err,
                    "train/win_recall": avg_win_recall,
                }
                if sprite_loss_weight > 0:
                    wandb_log["train/sprite_mse"] = avg_sprite_mse
                if use_vq:
                    wandb_log["vq/codebook_loss"] = avg_vq_cb
                    wandb_log["vq/commit_loss"] = avg_vq_commit
                    wandb_log["vq/codebook_utilization"] = avg_vq_util
                    wandb_log["vq/usage_loss"] = avg_vq_usage
                    wandb_log["vq/soft_perplexity"] = avg_vq_perp
                wandb.log(wandb_log, step=global_step)

            # Per-game diagnostic pass (multi-game only, with game_names known).
            if per_game_eval is not None and global_step % per_game_eval_interval == 0:
                worst = ("", 1.0, 0.0)  # (name, change_acc, loss) — lowest change_acc
                for g, idx_g in per_game_eval.items():
                    # idx_g = LOCAL indices into per_game_states[g]; pad to global max.
                    N_eval = len(idx_g)
                    C_g = int(game_CHW[g, 0]); H_g = int(game_CHW[g, 1]); W_g = int(game_CHW[g, 2])
                    s_eval = np.zeros((N_eval, max_C, max_H, max_W), dtype=np.uint8)
                    ns_eval = np.zeros((N_eval, max_C, max_H, max_W), dtype=np.uint8)
                    full_s = _unpack_states(per_game_states[g][idx_g], W_g)
                    full_ns = _unpack_states(per_game_next_states[g][idx_g], W_g)
                    eval_mask_np = np.zeros((N_eval, max_C, max_H, max_W), dtype=np.uint8)
                    for ii, j_local in enumerate(idx_g):
                        C_r, H_r, W_r = (int(x) for x in per_game_transition_shapes[g][j_local])
                        oy, _ = _pad_offsets(H_r, H_g)
                        ox, _ = _pad_offsets(W_r, W_g)
                        s_eval[ii, :C_r, :H_r, :W_r] = full_s[ii, :C_r, oy:oy + H_r, ox:ox + W_r]
                        ns_eval[ii, :C_r, :H_r, :W_r] = full_ns[ii, :C_r, oy:oy + H_r, ox:ox + W_r]
                        eval_mask_np[ii, :C_r, :H_r, :W_r] = 1
                    eval_mask = jnp.array(eval_mask_np, dtype=jnp.float32)
                    s_g = jnp.array(s_eval, dtype=jnp.float32)
                    a_int_g = per_game_actions_np[g][idx_g]
                    a_oh_g = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a_int_g])
                    ns_g = jnp.array(ns_eval, dtype=jnp.float32)
                    hs_g = ha_g = None
                    if hist_k > 0:
                        _hs, _ha = _gather_history(
                            [(g, int(j)) for j in idx_g],
                            max_C, max_H, max_W, np_rng)
                        hs_g = jnp.array(_hs, dtype=jnp.float32)
                        ha_g = jnp.array(_ha, dtype=jnp.int32)
                    if conditional:
                        gt_g = jnp.array(np.broadcast_to(tokens_np[g], (N_eval,) + tokens_np.shape[1:]))
                        gm_g = jnp.array(np.broadcast_to(masks_np[g], (N_eval,) + masks_np.shape[1:]))
                        pg_loss, pg_acc, pg_cacc = eval_forward(
                            params, s_g, a_oh_g, ns_g, gt_g, gm_g, eval_mask,
                            hist_states=hs_g, hist_actions=ha_g,
                        )
                    else:
                        pg_loss, pg_acc, pg_cacc = eval_forward(
                            params, s_g, a_oh_g, ns_g, eval_mask,
                            hist_states=hs_g, hist_actions=ha_g,
                        )
                    pg_loss, pg_acc, pg_cacc = float(pg_loss), float(pg_acc), float(pg_cacc)
                    per_game_log[g].append({
                        "step": global_step, "loss": pg_loss,
                        "acc": pg_acc, "change_acc": pg_cacc,
                    })
                    if pg_cacc < worst[1]:
                        worst = (game_names[g], pg_cacc, pg_loss)
                    if wandb.run is not None:
                        name = game_names[g]
                        wandb.log({
                            f"train/per_game/{name}/loss": pg_loss,
                            f"train/per_game/{name}/err": 1.0 - pg_acc,
                            f"train/per_game/{name}/change_err": 1.0 - pg_cacc,
                        }, step=global_step)
                if worst[0]:
                    print(f"    worst-fit game: {worst[0]}  "
                          f"change_err={1-worst[1]:.3e}  loss={worst[2]:.3e}")

            # === Held-out test eval ===
            # Per-game per-bucket pass over (a uniform sample of) the
            # transitions held out at dataset-load time. Same loss/acc
            # definitions as training, computed on data the model never
            # sees a gradient on. Empty when val_frac == 0.
            if (_val_bucket_to_game_indices and
                    global_step % val_eval_interval == 0):
                tot_loss = 0.0
                tot_acc = 0.0
                tot_cacc = 0.0
                tot_n = 0
                va_eye = np.eye(N_ACTIONS, dtype=np.float32)
                for (H_b, W_b), games_d in _val_bucket_to_game_indices.items():
                    # Gather (game, local_idx) pairs across games in this bucket.
                    pairs: list[tuple[int, int]] = []
                    for g_va, idx_pool in games_d.items():
                        if len(idx_pool) > val_eval_max_per_game:
                            sub_rng = np.random.RandomState(seed * 31337 + g_va + global_step)
                            idx_pool = sub_rng.choice(idx_pool,
                                                      size=val_eval_max_per_game,
                                                      replace=False)
                        pairs.extend((g_va, int(j)) for j in idx_pool)
                    if not pairs:
                        continue
                    # Process in chunks of batch_size to bound peak memory.
                    for chunk_start in range(0, len(pairs), batch_size):
                        chunk = pairs[chunk_start:chunk_start + batch_size]
                        nb = len(chunk)
                        s_buf_va  = np.zeros((nb, max_C, H_b, W_b), dtype=np.uint8)
                        ns_buf_va = np.zeros((nb, max_C, H_b, W_b), dtype=np.uint8)
                        mask_buf_va = np.zeros((nb, max_C, H_b, W_b), dtype=np.uint8)
                        a_buf_va = np.zeros((nb,), dtype=np.int32)
                        g_buf_va = np.zeros((nb,), dtype=np.int32)
                        for i, (g_va, j_va) in enumerate(chunk):
                            W_store = int(game_CHW[g_va, 2])
                            C_r, H_r, W_r = (int(x) for x in
                                             per_game_transition_shapes[g_va][j_va])
                            s_full = _unpack_states(per_game_states[g_va][j_va], W_store)
                            ns_full = _unpack_states(per_game_next_states[g_va][j_va], W_store)
                            oy, _ = _pad_offsets(H_r, s_full.shape[1])
                            ox, _ = _pad_offsets(W_r, W_store)
                            s_buf_va[i,  :C_r, :H_r, :W_r] = s_full[:C_r,  oy:oy + H_r, ox:ox + W_r]
                            ns_buf_va[i, :C_r, :H_r, :W_r] = ns_full[:C_r, oy:oy + H_r, ox:ox + W_r]
                            mask_buf_va[i, :C_r, :H_r, :W_r] = 1
                            a_buf_va[i] = per_game_actions_np[g_va][j_va]
                            g_buf_va[i] = g_va
                        s_va = jnp.array(s_buf_va, dtype=jnp.float32)
                        ns_va = jnp.array(ns_buf_va, dtype=jnp.float32)
                        a_oh_va = jnp.array(va_eye[a_buf_va])
                        spatial_mask_va = jnp.array(mask_buf_va, dtype=jnp.float32)
                        hs_va = ha_va = None
                        if hist_k > 0:
                            _hs, _ha = _gather_history(
                                chunk, max_C, H_b, W_b, np_rng)
                            hs_va = jnp.array(_hs, dtype=jnp.float32)
                            ha_va = jnp.array(_ha, dtype=jnp.int32)
                        if conditional:
                            gt_va = jnp.array(tokens_np[g_buf_va])
                            gm_va = jnp.array(masks_np[g_buf_va])
                            va_loss, va_acc, va_cacc = eval_forward(
                                params, s_va, a_oh_va, ns_va, gt_va, gm_va,
                                spatial_mask_va,
                                hist_states=hs_va, hist_actions=ha_va,
                            )
                        else:
                            va_loss, va_acc, va_cacc = eval_forward(
                                params, s_va, a_oh_va, ns_va, spatial_mask_va,
                                hist_states=hs_va, hist_actions=ha_va,
                            )
                        # Weight by chunk size so the average is per-transition.
                        tot_loss += float(va_loss) * nb
                        tot_acc += float(va_acc) * nb
                        tot_cacc += float(va_cacc) * nb
                        tot_n += nb
                if tot_n > 0:
                    va_l = tot_loss / tot_n
                    va_a = tot_acc / tot_n
                    va_c = tot_cacc / tot_n
                    val_log.append({
                        "step": int(global_step),
                        "loss": va_l,
                        "acc": va_a,
                        "change_acc": va_c,
                        "n_transitions": int(tot_n),
                    })
                    print(f"    val (held-out, n={tot_n:,}):  loss={va_l:.4e}  "
                          f"err={1-va_a:.4e}  change_err={1-va_c:.4e}")
                    if wandb.run is not None:
                        wandb.log({
                            "val/loss": va_l,
                            "val/err": 1.0 - va_a,
                            "val/change_err": 1.0 - va_c,
                        }, step=global_step)

            # Intermittent rollout GIF (true vs pred side-by-side)
            if gif_enabled and global_step % gif_interval == 0 and global_step > 0:
                _maybe_render_gif(global_step, avg_cerr=avg_cerr)

            # Periodically save curves so plot mode can pick up in-progress runs
            if global_step % 10_000 == 0:
                save_data = {
                    "losses": np.array(losses),
                    "accs": np.array(accs),
                    "change_accs": np.array(change_accs),
                    "vq_cb_losses": np.array(vq_cb_losses),
                    "vq_commit_losses": np.array(vq_commit_losses),
                    "vq_utils": np.array(vq_utils),
                    "vq_usage_losses": np.array(vq_usage_losses),
                    "vq_soft_perplexities": np.array(vq_soft_perplexities),
                }
                if per_game_log:
                    for g, rows in per_game_log.items():
                        if not rows:
                            continue
                        name = game_names[g]
                        save_data[f"per_game_{name}_step"] = np.array([r["step"] for r in rows])
                        save_data[f"per_game_{name}_loss"] = np.array([r["loss"] for r in rows])
                        save_data[f"per_game_{name}_acc"] = np.array([r["acc"] for r in rows])
                        save_data[f"per_game_{name}_change_acc"] = np.array(
                            [r["change_acc"] for r in rows])
                if val_log:
                    save_data["val_step"] = np.array([r["step"] for r in val_log])
                    save_data["val_loss"] = np.array([r["loss"] for r in val_log])
                    save_data["val_acc"] = np.array([r["acc"] for r in val_log])
                    save_data["val_change_acc"] = np.array(
                        [r["change_acc"] for r in val_log])
                    save_data["val_n_transitions"] = np.array(
                        [r["n_transitions"] for r in val_log])
                np.savez(
                    os.path.join(save_dir, f"curves_step{global_step}.npz"),
                    **save_data,
                )

            # Best-loss checkpoint (saved separately from the "latest"
            # checkpoint, so we don't lose the best params when the model
            # walks out of a sharp minimum). Save on ANY improvement.
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "params_best.pkl")
                tmp = best_path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(jax.device_get(params), f)
                os.replace(tmp, best_path)
                meta_path = os.path.join(save_dir, "train_meta.json")
                meta = {}
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    except Exception:
                        meta = {}
                meta["best_loss"] = float(best_loss)
                meta["best_step"] = int(global_step)
                tmp_meta = meta_path + ".tmp"
                with open(tmp_meta, "w") as f:
                    json.dump(meta, f, indent=2)
                os.replace(tmp_meta, meta_path)

            # Early stopping: patience ticks when loss hasn't improved by
            # min_delta vs the last "reset" reference. This keeps patience
            # independent of best_loss tracking (which updates on any
            # improvement).
            if patience > 0:
                if avg_loss < patience_ref_loss - min_delta:
                    patience_ref_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at step {global_step:,}: loss "
                          f"has not improved by {min_delta} for "
                          f"{patience} eval windows ({patience * log_interval:,} steps). "
                          f"Best loss={best_loss:.4e} @ step ~{meta.get('best_step', '?')}")
                    early_stopped = True
                    break

    # Final atomic checkpoint
    final_step = start_step + (step + 1) if early_stopped else start_step + n_updates
    _atomic_save_checkpoint(save_dir, params, final_step,
                             early_stopped=early_stopped,
                             n_updates_requested=start_step + n_updates)
    ckpt_path = os.path.join(save_dir, "params.pkl")
    print(f"Saved params to {ckpt_path} (total steps: {final_step:,})")

    # Final curves save, including per-game arrays (the main() caller's save is
    # dropped in favor of this to avoid clobbering per-game data).
    save_data = {
        "losses": np.array(losses),
        "accs": np.array(accs),
        "change_accs": np.array(change_accs),
        "vq_cb_losses": np.array(vq_cb_losses),
        "vq_commit_losses": np.array(vq_commit_losses),
        "vq_utils": np.array(vq_utils),
        "vq_usage_losses": np.array(vq_usage_losses),
        "vq_soft_perplexities": np.array(vq_soft_perplexities),
    }
    if per_game_log:
        for g, rows in per_game_log.items():
            if not rows:
                continue
            name = game_names[g]
            save_data[f"per_game_{name}_step"] = np.array([r["step"] for r in rows])
            save_data[f"per_game_{name}_loss"] = np.array([r["loss"] for r in rows])
            save_data[f"per_game_{name}_acc"] = np.array([r["acc"] for r in rows])
            save_data[f"per_game_{name}_change_acc"] = np.array(
                [r["change_acc"] for r in rows])
    if val_log:
        save_data["val_step"] = np.array([r["step"] for r in val_log])
        save_data["val_loss"] = np.array([r["loss"] for r in val_log])
        save_data["val_acc"] = np.array([r["acc"] for r in val_log])
        save_data["val_change_acc"] = np.array(
            [r["change_acc"] for r in val_log])
        save_data["val_n_transitions"] = np.array(
            [r["n_transitions"] for r in val_log])
    np.savez(os.path.join(save_dir, f"curves_step{final_step}.npz"), **save_data)

    return params, losses, accs, change_accs, final_step


# ---------------------------------------------------------------------------
# 4. Evaluation — multi-step rollout with the learned world model
# ---------------------------------------------------------------------------

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


# Module-level cache of JIT'd scan functions, keyed by (id(model), conditional).
# Hoisting these out of `_run_eval_rollouts_jax` is important: defining them
# inside the function would create fresh closures (and fresh JIT caches) on
# every call — so a Python loop over (game, level, mode) would recompile each
# time and JAX would be slower than the unbatched path.
_JAX_EVAL_CACHE: dict = {}


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


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train NCA world model on a PuzzleScript game")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--game", help="Single game name (e.g. pipe_bend, sokoban_basic)")
    g.add_argument("--games", help="Comma-separated game names, or a preset name (e.g. 'small')")
    g.add_argument("--n_per_rule_games", type=int, default=None,
                   help="Select the first N games sorted by (n_rules asc, "
                        "name asc) from --n_per_rule_universe (default: dedup_pool, "
                        "with max_level_area<=30 filter and Heldout-26 exclusion). "
                        "Mutually exclusive with --game/--games.")
    p.add_argument("--n_per_rule_universe", choices=("dedup_pool", "gallery"),
                   default="dedup_pool",
                   help="Where --n_per_rule_games draws from. dedup_pool (default): "
                        "all 3,474 token-deduped games at data/dedup_candidates_v2.json "
                        "(2,447 games at n_rules<=10 vs gallery's 22). gallery: legacy "
                        "subset of curated games.")
    p.add_argument("--n_per_rule_max_area", type=int, default=30,
                   help="max_level_area cap on --n_per_rule_games selection. Defaults "
                        "to 30 to match canonical Train-X presets and avoid OOM on "
                        "large grids (a single 64x64 game balloons activation memory "
                        "to >20 GiB on n_hid=256, n_nca_steps=8, with all pool features).")
    p.add_argument("--n_per_rule_include_random", action=argparse.BooleanOptionalAction, default=False,
                   help="Include games with stochastic dynamics (`random`/`randomDir` "
                        "rules) in --n_per_rule_games selection. Default False because "
                        "stochastic games have an irreducible val_cerr floor (no model "
                        "can perfectly predict random outcomes), which biases the "
                        "aggregate downward and obscures the cond-vs-uncond comparison. "
                        "Source of truth: `has_randomness` field in games_metadata.json "
                        "(populated by the JS engine's compile pass).")
    p.add_argument("--n_per_rule_strategy", choices=("sorted", "stratified"),
                   default="sorted",
                   help="How --n_per_rule_games selects from the pool. 'sorted' "
                        "(default, legacy): take first N from (n_rules asc, name asc) "
                        "ranked list — yields rule-complexity-truncated training set "
                        "(at n=800 the max n_rules is only 3, vs Heldout-26 median 6). "
                        "'stratified': bucket by n_rules in [1..n_per_rule_max_bucket], "
                        "take ceil(N/n_buckets) from each bucket sorted by name, then "
                        "trim to N. This covers higher-complexity games at the cost "
                        "of training-time A* on slower games.")
    p.add_argument("--n_per_rule_max_bucket", type=int, default=10,
                   help="For --n_per_rule_strategy=stratified: cap the highest n_rules "
                        "bucket. Bucket K means n_rules>=K are pooled together. "
                        "Default 10 covers ~83%% of Heldout-26 (median 6, 90th pct 14).")
    p.add_argument("--level", type=int, default=None,
                   help="Train on a single level index. Default: all levels.")
    p.add_argument("--train_levels", default=None,
                   help="Comma-separated level indices to train on (e.g. '0,1,8'). "
                        "Overrides --level when set. Eval still runs over all "
                        "levels per-game so held-out levels are reported as "
                        "out-of-distribution metrics.")
    # Data collection (search-driven unique-transition exploration; see
    # collect_unique_transitions). --search_algos (plural) is for evaluation
    # rollouts only, not training data collection.
    p.add_argument("--n_search_steps", type=int, default=100_000)
    p.add_argument("--search_timeout_ms", type=int, default=-1)
    p.add_argument("--search_algos", nargs="+", default=["bfs", "astar"],
                   help="Algorithms used during evaluation/GIF rollouts.")
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--search_algo", default="astar", choices=["astar", "bfs"],
                   help="Search algorithm for training-data collection (default: astar)")
    # Architecture
    p.add_argument("--n_nca_steps", type=int, default=4,
                   help="Number of NCA update steps per forward pass")
    p.add_argument("--n_hid", type=int, default=256)
    p.add_argument("--history", type=int, default=0,
                   help="Number of preceding (state, action) transitions to "
                        "feed as extra input channels alongside the current "
                        "one (Option A). 0 (default) = current transition "
                        "only, byte-identical to the no-history model. >0 "
                        "requires contiguous trajectory windows from the data "
                        "pipeline; intended for in-context dynamics inference.")
    p.add_argument("--cond_mask_prob", type=float, default=0.0,
                   help="Classifier-free-guidance-style rule dropout: per "
                        "training example, probability of withholding the rule "
                        "encoding (FiLM z / dynamics slots zeroed). 0 (default) "
                        "= always conditioned, identical to before. >0 trains "
                        "one set of weights as both the conditional model and "
                        "its rule-marginal; eval always uses full conditioning. "
                        "No effect on unconditional (--no-conditional) runs.")
    # Conditional model
    p.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ConditionalNCAWorldModel with game-spec encoder. "
                        "Pass --no-conditional for the unconditional baseline.")
    p.add_argument("--d_z", type=int, default=64, help="Latent dimension for game encoder")
    p.add_argument("--d_model", type=int, default=64, help="Transformer hidden dim")
    p.add_argument("--n_enc_layers", type=int, default=2, help="Transformer encoder layers")
    p.add_argument("--n_heads", type=int, default=4, help="Transformer attention heads")
    p.add_argument("--architecture", type=str, default="rule_attn",
                   choices=["film", "rule_attn", "cnn", "unet", "vit"],
                   help="Spatial-update body. 'rule_attn' (default) = NCA "
                        "with per-step cross-attention to K rule slots. "
                        "'film' = NCA with pooled-z FiLM conditioning. "
                        "Non-NCA baselines (share the rule_attn slot encoder "
                        "but apply a one-shot body): 'cnn' = deep ResNet, "
                        "'unet' = 2-level U-Net with bottleneck FiLM, 'vit' "
                        "= Transformer over flattened cells with slot "
                        "cross-attn. Baselines do not support --adaptive_halt "
                        "or --vq_codebook.")
    p.add_argument("--baseline_n_blocks", type=int, default=4,
                   help="Number of CNN-baseline residual blocks "
                        "(--architecture cnn). Param count scales with this.")
    p.add_argument("--baseline_n_levels", type=int, default=2,
                   help="Number of U-Net down/up levels (--architecture unet).")
    p.add_argument("--baseline_n_layers", type=int, default=4,
                   help="Number of ViT encoder layers (--architecture vit).")
    p.add_argument("--n_slots", type=int, default=16,
                   help="Number of rule slots for --architecture rule_attn.")
    p.add_argument("--d_slot", type=int, default=64,
                   help="Per-slot dim for --architecture rule_attn.")
    p.add_argument("--n_app_slots", type=int, default=1,
                   help="Of the --n_slots, how many are appearance-only "
                        "(decoder sees, NCA does not). Encourages dynamics "
                        "and visual info to occupy disjoint slot subsets.")
    # VQ-VAE-style codebook on the encoder slots. Off by default — enabling
    # adds a `slot_vq/codebook` param and two VQ losses; default-off path is
    # parameter-identical to the pre-VQ model so existing checkpoints load.
    p.add_argument("--vq_codebook", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="If set, quantize encoder slots to a learned shared "
                        "codebook (VQ-VAE style) before they enter the NCA. "
                        "Only valid with --architecture rule_attn.")
    p.add_argument("--vq_codebook_size", type=int, default=512,
                   help="Number of entries in the slot codebook.")
    p.add_argument("--vq_commitment_weight", type=float, default=0.25,
                   help="Beta in vq_total = codebook_loss + beta*commitment_loss.")
    p.add_argument("--vq_loss_weight", type=float, default=1.0,
                   help="Multiplier on vq_total when added to the training loss.")
    p.add_argument("--vq_usage_loss_weight", type=float, default=0.0,
                   help="Multiplier on a differentiable soft-assignment "
                        "entropy penalty that discourages codebook collapse. "
                        "0.0 preserves historical VQ behavior.")
    p.add_argument("--vq_entropy_temp", type=float, default=1.0,
                   help="Temperature for the soft assignment distribution used "
                        "only by --vq_usage_loss_weight diagnostics/loss.")
    # Joint token-decoder training (encoder is shared with WM; decoder
    # cross-attends to ALL slots, while NCA only sees the dyn slots).
    p.add_argument("--token_decoder_loss_weight", type=float, default=0.0,
                   help="If >0, co-train an AR token decoder (recovering the "
                        "PuzzleScript source from the encoder slots) and add "
                        "this scaled cross-entropy to the WM loss.")
    p.add_argument("--decoder_d_model", type=int, default=128)
    p.add_argument("--decoder_n_layers", type=int, default=4)
    p.add_argument("--decoder_n_heads", type=int, default=4)
    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_updates", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--patience", type=int, default=300,
                   help="Early stopping patience (in eval windows of log_interval steps). "
                        "0 disables early stopping.")
    p.add_argument("--min_delta", type=float, default=1e-8,
                   help="Minimum change_acc improvement to reset patience counter")
    p.add_argument("--win_loss_weight", type=float, default=1.0,
                   help="Scalar multiplier on the win-prediction BCE term (0 disables the head's contribution to grads)")
    p.add_argument("--win_pos_weight", type=float, default=1.0,
                   help="Positive-class weight for the win BCE (raise to counter class imbalance; "
                        "a good rule of thumb is ~(#negatives / #positives))")
    p.add_argument("--ckpt_interval", type=int, default=1000,
                   help="Steps between periodic atomic saves of params.pkl during training. "
                        "Lower means a --render_only process sees fresher weights.")
    p.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=True,
                   help="Multi-game only: draw each batch with equal share per game "
                        "(replaces uniform-over-transitions sampling). Compensates "
                        "for per-game dataset-size imbalance. Pass --no-balanced_sampling "
                        "to fall back to uniform-over-transitions.")
    p.add_argument("--obj_permute_aug", action=argparse.BooleanOptionalAction, default=False,
                   help="Conditional only: per-batch random permutation of object "
                        "channels at training time. Permutes axis-1 of state/next-state/"
                        "spatial-mask tensors AND remaps CH<i> tokens in game_tokens "
                        "(and decoder target) to CH<perm[i]>. Forces the encoder to "
                        "represent rule structure rather than memorizing per-game "
                        "channel-to-object identities. Per-batch (one π for all samples) "
                        "for vmap efficiency; over training, every channel sees every "
                        "object role. Free at inference: π is identity at eval.")
    p.add_argument("--encode_sprites", action="store_true",
                   help="Include each object's palette + 5x5 sprite in the "
                        "game-spec token sequence (uses VOCAB_SIZE_EXT and a "
                        "larger max_seq_len). Required to decode visual games "
                        "from the latent space.")
    p.add_argument("--sprite_loss_weight", type=float, default=0.0,
                   help="Weight on a sprite-decoder MSE loss term. When >0, "
                        "adds a Dense head on z predicting each channel's "
                        "5x5x4 RGBA kernel; loss = MSE vs target sprite "
                        "from OBJECTS section. (Decoder head itself lands in "
                        "a follow-up; flag is plumbed for dataset-side prep.)")
    p.add_argument("--change_loss_weight", type=float, default=5.0,
                   help="Extra weight on changed cells in state BCE. 0 = uniform "
                        "mean. >0 = each changed cell counts "
                        "(1 + change_loss_weight)x vs unchanged. Default 5.0; "
                        "needed to escape identity collapse on multi-game sets "
                        "where most cells don't change between t and t+1.")
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["constant", "cosine"],
                   help="LR schedule. 'cosine' (default) anneals from --lr down "
                        "to --lr_min over n_updates steps. 'constant' keeps --lr.")
    p.add_argument("--lr_min", type=float, default=1e-7,
                   help="Floor LR for cosine schedule. Ignored for constant.")
    p.add_argument("--gif_interval", type=int, default=0,
                   help="Render an intermittent (real | pred) rollout GIF "
                        "every N training steps (also at step 0). 0 disables.")
    p.add_argument("--gif_n_steps", type=int, default=15,
                   help="Length of each intermittent training GIF rollout.")
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Clip gradients by global norm to this value (0 = off).")
    p.add_argument("--use_layernorm", action="store_true",
                   help="Apply a shared LayerNorm on h between NCA steps "
                        "(post-step on FiLM/uncond, pre-step on rule_attn). "
                        "Stabilizes deep unrolls (large n_nca_steps).")
    p.add_argument("--input_skip", action="store_true",
                   help="rule_attn-only: re-inject the embedded (state, "
                        "action) input into the conv at every NCA step. "
                        "Mirrors the input-skip in NCAWorldModel; helps "
                        "deep unrolls keep contact with the original "
                        "observation. No-op for FiLM/uncond (those already "
                        "have the input skip).")
    p.add_argument("--adaptive_halt", action="store_true",
                   help="rule_attn-only: enable PonderNet-style adaptive "
                        "halting. The model emits a halting probability at "
                        "every NCA step from a small head over pooled `h`. "
                        "Loss is the expected per-step heads-loss under the "
                        "induced halt distribution, plus a KL-to-geometric "
                        "prior regularizer (--halt_prior_p / --halt_kl_weight). "
                        "Requires --n_nca_repeats=n_nca_steps for the "
                        "halt-at-step-k semantic to be coherent (one shared "
                        "rule layer applied k times); otherwise different "
                        "halts pick between distinct per-step rule layers. "
                        "Currently supported only in the conditional, "
                        "non-VQ, non-joint-decoder training path.")
    p.add_argument("--halt_prior_p", type=float, default=0.1,
                   help="Geometric-prior parameter for adaptive halt. "
                        "Smaller → encourages later halt (longer rollouts). "
                        "Default 0.1 → expected ~10 steps under the prior.")
    p.add_argument("--halt_kl_weight", type=float, default=0.01,
                   help="Weight on KL(p || Geom(halt_prior_p)) in the "
                        "ponder loss. Larger → stronger pressure toward the "
                        "geometric prior; smaller → halt distribution is "
                        "fit to data with less regularization. Ignored "
                        "when --halt_mode=uniform.")
    p.add_argument("--halt_mode",
                   choices=["ponder", "uniform", "argmax_st", "convergence_st"],
                   default="ponder",
                   help="Per-step loss aggregation under --adaptive_halt. "
                        "'ponder' (default) is PonderNet-style with the "
                        "learned halt distribution + KL prior. 'uniform' "
                        "weights every step equally — pre-requisite for "
                        "convergence-based stopping at inference, since "
                        "the body has to be good at every depth (not just "
                        "at the expected halt step). Set --halt_kl_weight=0 "
                        "with uniform. 'argmax_st' uses straight-through "
                        "estimator on the argmax of p: forward computes L "
                        "only at the single selected step, backward updates "
                        "halt logits via soft p. Removes the shortcut "
                        "pressure of weighing every k, but body specialises "
                        "at one depth per example — keep KL prior on to "
                        "prevent halt collapse to k=1. 'convergence_st' "
                        "selects the first k at which the binary readout "
                        "has converged (fraction of cells flipping below "
                        "halt_prior_p), then computes L only at that step. "
                        "No halt-head signal is used; body gradient flows "
                        "through k* iterations only. Training and inference "
                        "share the same halting rule mechanically.")
    p.add_argument("--n_nca_repeats", type=int, default=1,
                   help="rule_attn-only: factor n_nca_steps into a "
                        "(n_layers × n_repeats) hierarchy mirroring the "
                        "PuzzleScript engine's two loop levels. Inner block "
                        "of n_layers = n_nca_steps // n_nca_repeats distinct "
                        "rule-application layers (each with its own weights, "
                        "analogous to one ordered rule list). Outer loop "
                        "applies the inner block n_nca_repeats times, sharing "
                        "weights across repeats — analogous to the engine's "
                        "`again` loop. Defaults (n_nca_repeats=1) reproduce "
                        "the historical per-step body bit-identically; "
                        "n_nca_repeats=n_nca_steps is fully shared (one rule "
                        "layer applied n_steps times). Constraint: "
                        "n_nca_steps must be divisible by n_nca_repeats. "
                        "No-op for FiLM/uncond.")
    p.add_argument("--max_transitions_per_game", type=int, default=200_000,
                   help="Cap per-game transition count (uniformly subsample). "
                        "0 disables. Essential for scaling to many games with "
                        "disparate sizes — prevents dataset OOM.")
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="Fraction of each game's transitions held out as the "
                        "in-distribution test set. Held-out indices are picked "
                        "deterministically from `seed` and never sampled in "
                        "training batches. Default 0.0 = no holdout (legacy).")
    p.add_argument("--val_eval_interval", type=int, default=0,
                   help="Steps between held-out test-set evals; 0 (default) ties "
                        "the cadence to --log_interval.")
    # Architectural pool flags (see _pool_features). Independent booleans;
    # any combination may be active.
    p.add_argument("--axis_pool", action=argparse.BooleanOptionalAction, default=True,
                   help="Inject row-max + col-max pooled features at each NCA step "
                        "(handles `[ X | ... | Y ]` style rules — X/Y in same row/col).")
    p.add_argument("--axis_cummax", action=argparse.BooleanOptionalAction, default=True,
                   help="Inject directional cumulative-max (L→R, R→L, T→B, B→T) at "
                        "each NCA step (more expressive variant of axis_pool).")
    p.add_argument("--global_pool", action=argparse.BooleanOptionalAction, default=True,
                   help="Inject grid-global max-pool features at each NCA step "
                        "(handles `[X] [Y]` multi-bracket rules — X and Y both "
                        "exist somewhere on the level).")
    # Output
    # Synthetic-level generation. When --synthetic_levels > 0, replace the
    # authored levels of each requested game with N procedurally-generated
    # valid levels. The generator is game-agnostic: tile patterns are sampled
    # from the empirical distribution observed in the game's authored levels,
    # then exactly one player is forced. Validity = (not already winning,
    # solvable within BFS budget, ≥ min_states reachable, no timeout).
    p.add_argument("--synthetic_levels", type=int, default=0,
                   help="If >0, generate N synthetic levels instead of using authored levels")
    p.add_argument("--synthetic_w", type=int, default=7, help="Width of synthetic levels")
    p.add_argument("--synthetic_h", type=int, default=7, help="Height of synthetic levels")
    p.add_argument("--synthetic_seed", type=int, default=0, help="Seed for synthetic level generation")
    p.add_argument("--synthetic_min_states", type=int, default=20,
                   help="Min reachable BFS states for a synthetic level to be accepted")
    p.add_argument("--synthetic_mode", type=str, default="tile_pattern_empirical",
                   choices=["tile_pattern_empirical", "tile_pattern_uniform", "evolve"],
                   help="Level-finding strategy. 'tile_pattern_*' = rejection sampling "
                        "from the per-tile pattern distribution; 'evolve' = population GA "
                        "with BFS-iterations fitness (use for harder games where rejection "
                        "sampling has very low acceptance).")
    p.add_argument("--synthetic_per_game_size", action="store_true",
                   help="Auto-detect per-game synth grid size from each game's authored max-dim. "
                        "Empirically (see RUNNING_REPORT) only synth at the authored max-dim "
                        "transfers cleanly to authored levels — smaller misses rule structure, "
                        "bigger learns position-padding artifacts. Overrides --synthetic_w/h.")
    p.add_argument("--synthetic_multi_grid", action="store_true",
                   help="Generate at multiple grid sizes per game and merge via spatial padding. "
                        "Closes the residual gap on games with mixed authored sizes. Combined with "
                        "--synthetic_grid_sizes for explicit size list, or default uses authored sizes.")
    p.add_argument("--synthetic_grid_sizes", type=str, default=None,
                   help='Explicit comma-separated grid sizes for multi_grid: "5x5,7x7,9x9". '
                        'Overrides authored-size detection — recommended since authored sizes '
                        'can be very large (TSP 20x19, kettle 15x15) and synth at those defeats '
                        'the purpose of avoiding big-grid BFS. Hardcoded small sizes train at '
                        'tractable BFS depths and rely on size-up generalization to bigger eval levels.')
    p.add_argument("--synthetic_fallback_dynamics", action="store_true",
                   help="If a game produces 0 levels with require_solvable=True (e.g. Zen at "
                        "small grids), retry once with require_solvable=False so the multi-game "
                        "pipeline still gets dynamics-only data for that game.")
    p.add_argument("--synthetic_no_a_count_max", type=int, default=3,
                   help="For 'no A' (num=-1) win conditions, cap count(A) at start to this value "
                        "so BFS reachable-state-space stays tractable. Default 3 enables Zen-class "
                        "synth gen; sweep over {3,5,8,12} to balance solvability vs distribution match.")
    p.add_argument("--synthetic_evolve_pop_size", type=int, default=64)
    p.add_argument("--synthetic_evolve_max_generations", type=int, default=200)
    p.add_argument("--synthetic_evolve_n_mutations_min", type=int, default=1)
    p.add_argument("--synthetic_evolve_n_mutations_max", type=int, default=3)
    p.add_argument("--synthetic_seed_from_authored", action="store_true",
                   help="Seed the evolve population with cropped authored levels (instead "
                        "of random init). Only meaningful when --synthetic_mode=evolve.")
    p.add_argument("--synthetic_seed_level_indices", type=str, default=None,
                   help="Comma-separated authored-level indices used as evolve seeds "
                        "(e.g. '0' or '0,1,2,3,4,5,6,7'). Default = all authored levels. "
                        "Only meaningful with --synthetic_seed_from_authored.")
    p.add_argument("--synthetic_evolve_selection",
                   choices=("fitness", "nslc"), default="fitness",
                   help="Evolve selection mode. 'fitness' (default): top-K elites by "
                        "raw fitness. 'nslc': novelty search with local competition — "
                        "elites are the non-dominated front in (novelty, local_competition) "
                        "space, where novelty = mean Hamming distance over the dat to k "
                        "nearest neighbours in (pop union archive), and local_competition "
                        "= how many of those neighbours the candidate beats on fitness. "
                        "Pressures the GA for structural diversity in addition to raw "
                        "BFS-depth/coverage fitness.")
    p.add_argument("--synthetic_nslc_k", type=int, default=5,
                   help="kNN count for NSLC novelty/local-competition (default 5).")
    p.add_argument("--synthetic_nslc_archive_size", type=int, default=500,
                   help="Cap on the NSLC novelty archive; oldest entries drop first.")
    p.add_argument("--synthetic_track_rules_fired", action="store_true",
                   help="Collect per-step rule-firing telemetry from the engine while "
                        "evolving levels. Required to activate either of the two "
                        "rule-coverage knobs below; cheap on its own (just adds "
                        "rules_fired_union to each cached level's payload).")
    p.add_argument("--synthetic_rule_coverage_weight", type=float, default=0.0,
                   help="GA fitness becomes `BFS_iterations + w * unique_rules_fired`. "
                        "Default 0 keeps the validated iterations-only fitness; set to "
                        "~50–100 to bias the GA toward levels that exercise more distinct "
                        "rules. Implies --synthetic_track_rules_fired.")
    p.add_argument("--synthetic_coverage_select_topk", action="store_true",
                   help="At the end of evolution, greedily pick the n_target levels "
                        "that maximize the *union* of rules fired across the dataset, "
                        "rather than the top-K individuals by fitness. Best at small "
                        "K (e.g. K=64) on multi-bracket games where individual-best "
                        "selection can pick up many copies of one mechanic. Implies "
                        "--synthetic_track_rules_fired.")
    p.add_argument("--synthetic_require_solvable", action=argparse.BooleanOptionalAction, default=True,
                   help="Reject levels with no winning transition observed within BFS budget. "
                        "Default True so the wons head sees positives; pass --no-synthetic_require_solvable "
                        "to keep all valid-dynamics levels (will produce wons=0 always, head will collapse).")
    p.add_argument("--synthetic_max_attempts_per_level", type=int, default=1000,
                   help="Total attempt budget = n_levels * this")
    p.add_argument("--synthetic_max_iters_search", type=int, default=5000,
                   help="BFS budget per synthetic level during validity check + transition collection")
    p.add_argument("--synthetic_timeout_ms_search", type=int, default=2000)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--render_gif", action="store_true", help="Render comparison GIF after training")
    p.add_argument("--render_only", action="store_true",
                   help="Skip data collection and training; load the latest checkpoint "
                        "from save_dir (or --load) and run eval + rendering only. "
                        "Use this alongside the same training args to inspect a "
                        "partially-trained model without disturbing the training run.")
    p.add_argument("--benchmark_eval", action="store_true",
                   help="Time the three random-rollout eval implementations "
                        "(per-episode loop / batched / JAX-scanned) on the first "
                        "(game, level) and exit. Use with --render_only.")
    p.add_argument("--load", default=None, metavar="DIR",
                   help="Load params from this dir instead of the default save_dir")
    p.add_argument("--play", action="store_true",
                   help="Interactive play mode (w/a/s/d/x keys)")
    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", default="nca-world-model", help="wandb project name")
    p.add_argument("--wandb_name", default=None, help="wandb run name (auto-generated if not set)")
    p.add_argument("--sweep_name", default=None,
                   help="Tag for grouping runs in sweep_nca_wm.py cross-evaluation")
    args = p.parse_args()
    parsed_train_levels = None
    if args.train_levels:
        parsed_train_levels = [int(x) for x in args.train_levels.split(",") if x.strip()]

    BASELINE_ARCHS = {"cnn", "unet", "vit"}
    if args.architecture in BASELINE_ARCHS:
        if not args.conditional:
            p.error(f"--architecture {args.architecture} requires --conditional "
                    "(baselines reuse the rule_attn slot encoder)")
        if args.vq_codebook:
            p.error(f"--vq_codebook is not supported with --architecture {args.architecture}")
        if args.adaptive_halt:
            p.error(f"--adaptive_halt is not supported with --architecture {args.architecture} "
                    "(no iteration → no halt semantics)")
    if args.vq_codebook and args.architecture != "rule_attn":
        p.error("--vq_codebook is only supported with --architecture rule_attn")
    if args.vq_codebook and not args.conditional:
        p.error("--vq_codebook requires --conditional (slots come from the "
                "game encoder, which only exists in conditional mode)")
    if args.adaptive_halt:
        if args.architecture != "rule_attn":
            p.error("--adaptive_halt is only supported with --architecture rule_attn")
        if not args.conditional:
            p.error("--adaptive_halt requires --conditional (the v1 ponder loss "
                    "is wired only into the conditional training path)")
        if args.vq_codebook:
            p.error("--adaptive_halt + --vq_codebook is not yet supported")
        if args.token_decoder_loss_weight > 0:
            p.error("--adaptive_halt + joint token decoder is not yet supported")
        if args.n_nca_repeats != args.n_nca_steps:
            p.error("--adaptive_halt requires --n_nca_repeats == --n_nca_steps "
                    f"(one shared rule layer applied n_nca_steps times); got "
                    f"n_nca_repeats={args.n_nca_repeats} vs n_nca_steps={args.n_nca_steps}")

    ps_parser = init_ps_lark_parser()
    multigame = args.games is not None or args.n_per_rule_games is not None

    if multigame:
        # --- Multi-game path ---
        if args.n_per_rule_games is not None:
            # Build the pool: either the dedup pool (default, broad coverage)
            # or the gallery (legacy, narrow). Filter by max_level_area to
            # avoid the single-large-grid OOM blowup, and exclude Heldout-26.
            heldout_names: set[str] = set()
            heldout_path = _REPO_ROOT / "data" / "heldout_v4_n30.json"
            if heldout_path.is_file():
                heldout_names = {h["name"] for h in
                                 json.loads(heldout_path.read_text())["heldout"]}

            # `has_randomness` lives in games_metadata.json (populated by the
            # JS engine's compile pass). Cross-lookup helper used by both
            # universe branches below.
            meta_path = _REPO_ROOT / "data" / "games_metadata.json"
            meta = json.loads(meta_path.read_text())
            def _meta_for(g: str) -> dict | None:
                for cand in (g + ".txt", g.replace(' ', '_') + ".txt",
                             g.lower() + ".txt"):
                    if cand in meta:
                        return meta[cand]
                return None

            ranked: list[tuple[int, str]] = []
            n_filtered_random = 0
            if args.n_per_rule_universe == "dedup_pool":
                dedup_path = _REPO_ROOT / "data" / "dedup_candidates_v2.json"
                pool = json.loads(dedup_path.read_text())["candidates"]
                for c in pool:
                    name = c["name"]
                    if name in heldout_names:
                        continue
                    n_rules = int(c.get("n_rules", -1))
                    if n_rules < 1:
                        continue
                    area = int(c.get("max_level_area", 999))
                    if area > args.n_per_rule_max_area:
                        continue
                    if not args.n_per_rule_include_random:
                        m = _meta_for(name)
                        if m is not None and m.get("has_randomness", False):
                            n_filtered_random += 1
                            continue
                    ranked.append((n_rules, name))
            else:  # "gallery"
                from puzzlescript_jax.utils import get_list_of_games_for_testing
                NCAWM_EXTRAS = ["nekopuzzle"]
                universe = list(get_list_of_games_for_testing(dataset="gallery"))
                for g in NCAWM_EXTRAS:
                    if g not in universe:
                        universe.append(g)
                for g in universe:
                    if g in heldout_names:
                        continue
                    m = _meta_for(g)
                    if m is None:
                        continue
                    n_rules = int(m.get("n_rules", -1))
                    if n_rules < 1:
                        continue
                    area = int(m.get("max_level_area", 999))
                    if area > args.n_per_rule_max_area:
                        continue
                    if (not args.n_per_rule_include_random
                            and m.get("has_randomness", False)):
                        n_filtered_random += 1
                        continue
                    ranked.append((n_rules, g))

            ranked.sort(key=lambda x: (x[0], x[1]))

            # The metadata `max_level_area` field stores only one dimension
            # (H), so a 29x65 game still slips through area<=30. Verify the
            # actual cached canvas (H from states.shape[1]; W from the npz
            # 'W' field) and reject games where any level exceeds the cap.
            # Only checks games already cached locally; un-cached games pass
            # this gate (they'll be cached and the next run will catch them).
            import glob as _glob
            cap = args.n_per_rule_max_area
            ranked_filtered: list[tuple[int, str]] = []
            n_filtered_canvas = 0
            for n_rules, name in ranked:
                files = sorted(_glob.glob(
                    f"rollout_data/{name}/level_*/astar_transitions_*.npz"))
                bad = False
                for fp in files:
                    try:
                        with np.load(fp, allow_pickle=True) as d:
                            H = int(d['states'].shape[1])
                            W = int(d['W'])
                            if H > cap or W > cap:
                                bad = True
                                break
                    except Exception:
                        pass  # corrupt/missing key — let it through
                if bad:
                    n_filtered_canvas += 1
                    continue
                ranked_filtered.append((n_rules, name))
            ranked = ranked_filtered

            if args.n_per_rule_strategy == "stratified":
                from collections import defaultdict as _dd, Counter as _Counter
                buckets: dict[int, list[tuple[int, str]]] = _dd(list)
                K = args.n_per_rule_max_bucket
                for n_rules, name in ranked:
                    if n_rules > K:
                        continue  # exclude games beyond cap (no pooling)
                    buckets[n_rules].append((n_rules, name))
                bucket_keys = sorted(buckets)
                per_bucket = max(1, args.n_per_rule_games // len(bucket_keys))
                picked: list[tuple[int, str]] = []
                for k in bucket_keys:
                    picked.extend(buckets[k][:per_bucket])
                if len(picked) < args.n_per_rule_games:
                    extras = []
                    for k in bucket_keys:
                        extras.extend(buckets[k][per_bucket:])
                    extras.sort(key=lambda x: (x[0], x[1]))
                    picked.extend(extras[:args.n_per_rule_games - len(picked)])
                picked = picked[:args.n_per_rule_games]
                game_names = [g for _, g in picked]
                _cnt = _Counter(p[0] for p in picked)
                _hist = " ".join(f"{nr}:{_cnt[nr]}" for nr in sorted(_cnt))
                rule_summary = (f"strat (K={K}): {len(bucket_keys)} buckets x "
                                f"{per_bucket}/bucket -> hist {_hist}")
            else:  # sorted
                game_names = [g for _, g in ranked[:args.n_per_rule_games]]
                rule_summary = (f"sorted: rules "
                                f"{ranked[0][0]}..{ranked[args.n_per_rule_games-1][0] if args.n_per_rule_games <= len(ranked) else ranked[-1][0]}")
            if not game_names:
                p.error("--n_per_rule_games selected zero games (empty universe "
                        "or all filtered out)")
            preset_tag = (f"nrules-{args.n_per_rule_games}-"
                          f"{args.n_per_rule_universe}-"
                          f"{args.n_per_rule_strategy}")
            random_note = (f", filtered {n_filtered_random} random-rule games"
                           if n_filtered_random > 0 else "")
            canvas_note = (f", filtered {n_filtered_canvas} oversize-canvas games"
                           if n_filtered_canvas > 0 else "")
            print(f"[--n_per_rule_games={args.n_per_rule_games} "
                  f"universe={args.n_per_rule_universe} max_area="
                  f"{args.n_per_rule_max_area} "
                  f"include_random={args.n_per_rule_include_random} "
                  f"strategy={args.n_per_rule_strategy}]: "
                  f"{len(game_names)} games selected from {len(ranked)} eligible"
                  f"{random_note}{canvas_note} ({rule_summary})")
        elif args.games == "gallery":
            # Full PuzzleScript gallery dataset via the shared helper.
            # Also add NCAWM-specific extras (games we reference a lot in
            # research but which aren't in games_dat.js or PRIORITY_GAMES).
            from puzzlescript_jax.utils import get_list_of_games_for_testing
            NCAWM_EXTRAS = ["nekopuzzle"]
            game_names = list(get_list_of_games_for_testing(dataset="gallery"))
            for g in NCAWM_EXTRAS:
                if g not in game_names:
                    game_names.append(g)
            preset_tag = "gallery"
        elif args.games in MULTI_GAME_PRESETS:
            game_names = MULTI_GAME_PRESETS[args.games]
            preset_tag = args.games
        else:
            game_names = [g.strip() for g in args.games.split(",")]
            preset_tag = f"{len(game_names)}games"

        # Tag only deviations from the canonical recipe; a default run gets a clean name.
        parts = []
        if not args.conditional: parts.append("uncond")
        if not args.balanced_sampling: parts.append("uniform")
        if not args.axis_pool: parts.append("no-ap")
        if not args.axis_cummax: parts.append("no-ac")
        if not args.global_pool: parts.append("no-gp")
        if args.encode_sprites: parts.append("spr")
        if args.change_loss_weight != 5.0: parts.append(f"clw{args.change_loss_weight:g}")
        if args.architecture != "rule_attn": parts.append(f"arch-{args.architecture}")
        if args.n_nca_repeats != 1: parts.append(f"rep{args.n_nca_repeats}")
        if args.adaptive_halt:
            if args.halt_mode == "uniform":
                parts.append("halt-uniform")
            else:
                parts.append(f"halt-p{args.halt_prior_p:g}-kl{args.halt_kl_weight:g}")
        if args.vq_codebook:
            parts.append(f"vq{args.vq_codebook_size}")
        if args.lr_schedule != "cosine": parts.append(f"lr-{args.lr_schedule}")
        if args.grad_clip != 0.5: parts.append(f"gc{args.grad_clip:g}")
        if args.synthetic_levels > 0:
            parts.append(
                f"synth{args.synthetic_levels}-{args.synthetic_w}x{args.synthetic_h}"
                f"-{args.synthetic_mode.replace('tile_pattern_', 'tp-')}"
                + ("-solv" if args.synthetic_require_solvable else "-any")
                + f"-s{args.synthetic_seed}"
            )
        recipe_tag = ("_" + "_".join(parts)) if parts else ""
        patience_tag = f"_pat-{args.patience}" if args.patience != 300 else ""
        save_dir = (args.save_dir or
                    f"nca_wm/logs/multi_{preset_tag}{recipe_tag}_level-{args.level}"
                    f"_nca-{args.n_nca_steps}_hid-{args.n_hid}_lr-{args.lr}"
                    f"{patience_tag}_s-{args.seed}")

        # Write config.json early so monitoring tools can see in-flight runs.
        # Overwritten verbatim at end of training (same content).
        # Also write RUNNING.pid early (before dataset load) so parallel
        # launchers' skip-check sees the lock before the next process
        # decides to launch. atexit cleanup removes it on exit.
        if not args.render_only:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            lock_path_early = os.path.join(save_dir, "RUNNING.pid")
            try:
                with open(lock_path_early, "w") as f:
                    f.write(str(os.getpid()))
                import atexit
                atexit.register(lambda: os.path.isfile(lock_path_early) and os.remove(lock_path_early))
            except Exception:
                pass

        if args.wandb:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_name or None,
                    config=vars(args),
                    dir=save_dir,
                    resume="allow",
                    settings=wandb.Settings(init_timeout=180),
                )
            except Exception as e:
                # Don't crash a multi-hour training run on a transient wandb
                # connection issue. Fall back to local-only logging.
                print(f"[wandb] init failed: {type(e).__name__}: {e}\n"
                      f"[wandb] continuing without wandb logging.")

        load_dir = args.load or save_dir
        if args.load is not None:
            load_cfg_path = os.path.join(load_dir, "config.json")
            if os.path.isfile(load_cfg_path):
                with open(load_cfg_path) as f:
                    load_cfg = json.load(f)
                load_vq = bool(load_cfg.get("vq_codebook", False))
                load_vq_size = int(load_cfg.get("vq_codebook_size", 512))
                if load_vq != bool(args.vq_codebook):
                    raise RuntimeError(
                        f"--load points to a run with vq_codebook={load_vq}, "
                        f"but this invocation has vq_codebook={args.vq_codebook}. "
                        "Pass matching VQ flags or use the original run command."
                    )
                if load_vq and load_vq_size != int(args.vq_codebook_size):
                    raise RuntimeError(
                        f"--load points to a VQ run with vq_codebook_size={load_vq_size}, "
                        f"but this invocation has vq_codebook_size={args.vq_codebook_size}."
                    )
        ckpt_path = os.path.join(load_dir, "params.pkl")
        infos_path = os.path.join(load_dir, "game_infos.pkl")

        # Load existing checkpoint + game_infos
        init_params = None
        start_step = 0
        if os.path.isfile(infos_path):
            with open(infos_path, "rb") as f:
                game_infos = pickle.load(f)
        else:
            game_infos = None

        if os.path.isfile(ckpt_path):
            print(f"Loading params from {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                init_params = pickle.load(f)
            meta_path = os.path.join(load_dir, "train_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    start_step = json.load(f).get("total_steps", 0)
            print(f"  Resuming from step {start_step:,}")

        # Collect/load dataset only if we need to train
        remaining = args.n_updates - start_step
        needs_training = remaining > 0
        if args.render_only:
            if init_params is None:
                raise RuntimeError(
                    f"--render_only: no checkpoint at {ckpt_path}. "
                    f"Wait for the training run to emit its first --ckpt_interval save."
                )
            if game_infos is None:
                raise RuntimeError(
                    f"--render_only: no game_infos.pkl at {infos_path}. "
                    f"The training run must have started and saved game_infos first."
                )
            needs_training = False
            print(f"--render_only: loaded checkpoint @ step {start_step:,}; "
                  f"skipping dataset collection and training.")

        if needs_training:
            names_to_collect = [g["name"] for g in game_infos] if game_infos else game_names
            os.makedirs(save_dir, exist_ok=True)
            if args.synthetic_levels > 0:
                dataset, game_infos = collect_multigame_dataset_synthetic(
                    names_to_collect, ps_parser,
                    n_levels=args.synthetic_levels,
                    width=args.synthetic_w,
                    height=args.synthetic_h,
                    seed=args.synthetic_seed,
                    mode=args.synthetic_mode,
                    require_solvable=args.synthetic_require_solvable,
                    max_attempts_per_level=args.synthetic_max_attempts_per_level,
                    max_iters_search=args.synthetic_max_iters_search,
                    timeout_ms_search=args.synthetic_timeout_ms_search,
                    min_states=args.synthetic_min_states,
                    per_game_size=args.synthetic_per_game_size,
                    multi_grid=args.synthetic_multi_grid,
                    grid_sizes=(
                        [tuple(int(x) for x in s.split("x"))
                         for s in args.synthetic_grid_sizes.split(",")]
                        if args.synthetic_grid_sizes else None
                    ),
                    fallback_dynamics=args.synthetic_fallback_dynamics,
                    no_a_count_max=args.synthetic_no_a_count_max,
                    track_rules_fired=(
                        args.synthetic_track_rules_fired
                        or args.synthetic_rule_coverage_weight != 0.0
                        or args.synthetic_coverage_select_topk
                    ),
                    rule_coverage_weight=args.synthetic_rule_coverage_weight,
                    coverage_select_topk=args.synthetic_coverage_select_topk,
                    evolve_pop_size=args.synthetic_evolve_pop_size,
                    evolve_max_generations=args.synthetic_evolve_max_generations,
                    evolve_n_mutations_min=args.synthetic_evolve_n_mutations_min,
                    evolve_n_mutations_max=args.synthetic_evolve_n_mutations_max,
                    seed_from_authored=args.synthetic_seed_from_authored,
                    seed_level_indices=(
                        [int(x) for x in args.synthetic_seed_level_indices.split(",")]
                        if args.synthetic_seed_level_indices else None
                    ),
                    selection=args.synthetic_evolve_selection,
                    nslc_k=args.synthetic_nslc_k,
                    nslc_archive_size=args.synthetic_nslc_archive_size,
                    encode_sprites=args.encode_sprites,
                    kernel_sep=getattr(args, "kernel_sep", False),
                    max_transitions_per_game=(args.max_transitions_per_game or None),
                )
            else:
                dataset, game_infos = collect_multigame_dataset(
                    names_to_collect, ps_parser,
                    level_i=args.level,
                    n_search_steps=args.n_search_steps,
                    search_timeout_ms=args.search_timeout_ms,
                    search_algo=args.search_algo,
                    encode_sprites=args.encode_sprites,
                    max_transitions_per_game=(args.max_transitions_per_game or None),
                    train_levels=parsed_train_levels,
                    val_frac=args.val_frac,
                )
            with open(infos_path, "wb") as f:
                pickle.dump(game_infos, f)
        elif game_infos is None:
            raise RuntimeError(
                f"No game_infos.pkl found at {infos_path} and no training to do. "
                "Run training first or provide --load pointing to a trained checkpoint."
            )

        max_C = max(g["n_objs"] for g in game_infos)
        pool_kwargs = dict(
            axis_pool=args.axis_pool,
            axis_cummax=args.axis_cummax,
            global_pool=args.global_pool,
            use_layernorm=args.use_layernorm,
        )
        use_sprite_dec = args.sprite_loss_weight > 0.0
        if args.conditional:
            max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
            max_tok_len = max(max_tok_len, 1)
            # Auto-size the vocab to cover only the tokens actually emitted
            # across the training set. V2 vocab IDs run up to VOCAB_SIZE_EXT_V2,
            # but games using fewer features only emit IDs in the lower range —
            # embedding those extra slots would just waste params. Floor at
            # VOCAB_SIZE_EXT + 1 since KERNEL_SEP (always emitted for multi-
            # kernel rules) sits at index VOCAB_SIZE_EXT.
            max_token_id_used = 0
            for g in game_infos:
                tids = g.get("token_ids") or []
                if tids:
                    max_token_id_used = max(max_token_id_used, int(max(tids)))
            vocab_size = max(max_token_id_used + 1, VOCAB_SIZE_EXT + 1)
            # Stash on args so subsequent vars(args) writes of config.json
            # carry vocab_size through to the saved config (loaders depend
            # on this field to rebuild the embedding).
            args.vocab_size = vocab_size
            if args.architecture == "rule_attn":
                from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
                model = RuleAttnNCAWorldModel(
                    n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C,
                    vocab_size=vocab_size + 1,
                    enc_d_model=args.d_model, enc_n_self_layers=args.n_enc_layers,
                    n_slots=args.n_slots, n_app_slots=args.n_app_slots,
                    d_slot=args.d_slot,
                    n_attn_heads=args.n_heads,
                    max_seq_len=max_tok_len + 1,
                    axis_pool=pool_kwargs.get("axis_pool", False),
                    axis_cummax=pool_kwargs.get("axis_cummax", False),
                    global_pool=pool_kwargs.get("global_pool", False),
                    use_vq=args.vq_codebook,
                    vq_codebook_size=args.vq_codebook_size,
                    vq_commitment_weight=args.vq_commitment_weight,
                    vq_entropy_temp=args.vq_entropy_temp,
                    use_layernorm=args.use_layernorm,
                    input_skip=args.input_skip,
                    n_repeats=args.n_nca_repeats,
                    adaptive_halt=args.adaptive_halt,
                    history=args.history,
                )
            elif args.architecture in ("cnn", "unet", "vit"):
                from nca_wm.baselines import (
                    CNNWorldModel, UNetWorldModel, ViTWorldModel,
                )
                shared_kw = dict(
                    n_hid=args.n_hid, n_out=max_C,
                    vocab_size=vocab_size + 1,
                    enc_d_model=args.d_model, enc_n_self_layers=args.n_enc_layers,
                    n_slots=args.n_slots, n_app_slots=args.n_app_slots,
                    d_slot=args.d_slot,
                    n_attn_heads=args.n_heads,
                    max_seq_len=max_tok_len + 1,
                    history=args.history,
                )
                if args.architecture == "cnn":
                    model = CNNWorldModel(
                        n_blocks=args.baseline_n_blocks,
                        axis_pool=pool_kwargs.get("axis_pool", False),
                        axis_cummax=pool_kwargs.get("axis_cummax", False),
                        global_pool=pool_kwargs.get("global_pool", False),
                        **shared_kw,
                    )
                elif args.architecture == "unet":
                    model = UNetWorldModel(
                        n_levels=args.baseline_n_levels,
                        **shared_kw,
                    )
                else:  # vit
                    model = ViTWorldModel(
                        n_layers=args.baseline_n_layers,
                        n_heads=args.n_heads,
                        **shared_kw,
                    )
            else:  # default: film
                model = ConditionalNCAWorldModel(
                    n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C,
                    vocab_size=vocab_size + 1,  # +1 for CLS
                    d_model=args.d_model, n_heads=args.n_heads,
                    n_enc_layers=args.n_enc_layers, d_z=args.d_z,
                    max_seq_len=max_tok_len + 1,  # +1 for CLS
                    sprite_decoder=use_sprite_dec,
                    history=args.history,
                    **pool_kwargs,
                )
        else:
            model = NCAWorldModel(
                n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C,
                input_skip=args.input_skip,
                n_repeats=args.n_nca_repeats,
                history=args.history,
                **pool_kwargs,
            )

        if needs_training:
            params, losses, accs, change_accs, final_step = train(
                model, dataset,
                lr=args.lr,
                n_updates=remaining,
                batch_size=args.batch_size,
                seed=args.seed,
                log_interval=args.log_interval,
                save_dir=save_dir,
                init_params=init_params,
                start_step=start_step,
                patience=args.patience,
                min_delta=args.min_delta,
                win_loss_weight=args.win_loss_weight,
                win_pos_weight=args.win_pos_weight,
                ckpt_interval=args.ckpt_interval,
                game_names=[g["name"] for g in game_infos],
                balanced_sampling=args.balanced_sampling,
                game_infos=game_infos,
                gif_interval=args.gif_interval,
                gif_n_steps=args.gif_n_steps,
                max_padded_shape=(
                    max_C,
                    max(g["H"] for g in game_infos),
                    max(g["W"] for g in game_infos),
                ),
                ps_parser=ps_parser,
                grad_clip=args.grad_clip,
                sprite_loss_weight=args.sprite_loss_weight,
                change_loss_weight=args.change_loss_weight,
                lr_schedule=args.lr_schedule,
                lr_min=args.lr_min,
                token_decoder_loss_weight=args.token_decoder_loss_weight,
                decoder_d_model=args.decoder_d_model,
                decoder_n_layers=args.decoder_n_layers,
                decoder_n_heads=args.decoder_n_heads,
                use_vq=args.vq_codebook,
                vq_commitment_weight=args.vq_commitment_weight,
                vq_loss_weight=args.vq_loss_weight,
                vq_usage_loss_weight=args.vq_usage_loss_weight,
                halt_prior_p=args.halt_prior_p,
                halt_kl_weight=args.halt_kl_weight,
                halt_mode=args.halt_mode,
                val_frac=args.val_frac,
                val_eval_interval=args.val_eval_interval,
                obj_permute_aug=args.obj_permute_aug,
                history=args.history,
                cond_mask_prob=args.cond_mask_prob,
            )
            # (curves saved inside train() with per-game arrays)
            os.makedirs(save_dir, exist_ok=True)
        else:
            if not args.render_only:
                print(f"Already at {start_step:,} steps (target {args.n_updates:,}), skipping training.")
            params = init_params
            final_step = start_step

        # Don't clobber config.json if another process is actively training
        if not args.render_only:
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

        # Downstream code expects WM-only params (eval/render don't use the
        # token decoder); unwrap if joint training produced a {wm,dec} dict.
        wm_params = _wm_p(params)

        if args.benchmark_eval:
            _benchmark_eval_impls(
                model, wm_params, game_infos,
                n_episodes=10, max_steps=50, rng_seed=0,
            )
            print("Benchmark done; skipping full eval.")
            if wandb.run is not None:
                wandb.finish()
            return

        # Per-game evaluation
        print("\nEvaluating per-game (autoregressive rollout)...")
        evaluate_multigame(
            model, wm_params, game_infos, ps_parser,
            search_algos=args.search_algos,
            search_n_steps=args.n_search_steps,
            search_timeout_ms=args.search_timeout_ms,
            save_dir=save_dir,
        )

        # Per-game GIFs (random + search, all levels). Gated on --render_gif
        # because rendering can take longer than training and blocks any sweep
        # that's queueing the next config. Render later with
        # `python train_nca_world_model.py ... --render_only --render_gif --load <dir>`.
        if args.render_gif:
            print("\nRendering per-game comparison GIFs...")
            render_multigame_gifs(
                model, wm_params, game_infos, ps_parser,
                save_dir=save_dir, step_label=final_step,
                search_algos=args.search_algos,
                search_n_steps=args.n_search_steps,
                search_timeout_ms=args.search_timeout_ms,
            )
        else:
            print("\nSkipping GIF rendering (pass --render_gif to render).")

        if wandb.run is not None:
            wandb.finish()
        print("Done!")
        return

    # --- Single-game path (original) ---
    # Single-game path defaults to level 0 for backwards compat
    if args.level is None:
        args.level = 0
    save_dir = (args.save_dir or
                f"nca_wm/logs/{args.game}_level-{args.level}_nca-{args.n_nca_steps}"
                f"_hid-{args.n_hid}_lr-{args.lr}_s-{args.seed}")

    # Compile game
    print(f"Compiling {args.game}...")
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, args.game)
    env = CppPuzzleScriptEnv(json_str, level_i=args.level, max_episode_steps=args.max_episode_steps)
    n_objs, H, W = env.observation_shape
    print(f"  obs_shape=({n_objs}, {H}, {W}), num_levels={env.num_levels}")

    model = NCAWorldModel(n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=n_objs,
                           axis_pool=args.axis_pool, axis_cummax=args.axis_cummax,
                           global_pool=args.global_pool, history=args.history)

    # Load existing checkpoint if available, otherwise train
    load_dir = args.load or save_dir
    ckpt_path = os.path.join(load_dir, "params.pkl")

    if args.render_only:
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(
                f"--render_only: no checkpoint at {ckpt_path}. "
                f"Wait for the training run to emit its first --ckpt_interval save."
            )
        print(f"--render_only: loading params from {ckpt_path} "
              f"(skipping dataset collection and training)")
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)
        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, args.game)
        print("\nEvaluating world model (autoregressive rollout)...")
        evaluate_world_model(model, params, json_str, level_i=args.level, save_dir=save_dir, history=args.history)
        print("\nRendering comparison GIFs...")
        render_post_training_gifs(
            model, params, json_str, backend_render,
            search_data=None, level_i=args.level,
            save_dir=save_dir,
        )
        print("Done!")
        return

    # Transition data for the post-training gif render (its search-action
    # overlay). Only available when we collect+train this run; None when params
    # are loaded from an existing checkpoint.
    search_data = None
    if os.path.isfile(ckpt_path):
        print(f"Loading params from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)
    else:
        print(f"Collecting unique transitions ({args.search_algo}, "
              f"{args.n_search_steps:,} iters / {args.search_timeout_ms:,}ms)...")
        dataset = collect_unique_transitions(
            json_str, args.game, level_i=args.level,
            max_iters=args.n_search_steps,
            timeout_ms=args.search_timeout_ms,
            search_algo=args.search_algo,
        )
        search_data = dataset
        print(f"Total dataset: {len(dataset['states']):,} transitions")

        # How many transitions actually involve a state change?
        changed = (dataset["states"] != dataset["next_states"]).any(axis=(1, 2, 3))
        print(f"  Transitions with state change: {changed.sum():,}/{len(changed):,} "
              f"({100*changed.mean():.1f}%)")

        params, losses, accs, change_accs, _ = train(
            model, dataset,
            lr=args.lr,
            n_updates=args.n_updates,
            batch_size=args.batch_size,
            seed=args.seed,
            log_interval=args.log_interval,
            save_dir=save_dir,
            patience=args.patience,
            min_delta=args.min_delta,
            win_loss_weight=args.win_loss_weight,
            win_pos_weight=args.win_pos_weight,
            ckpt_interval=args.ckpt_interval,
            val_frac=args.val_frac,
            val_eval_interval=args.val_eval_interval,
            obj_permute_aug=args.obj_permute_aug,
            history=args.history,
            cond_mask_prob=args.cond_mask_prob,
        )

        # Save training curves
        np.savez(
            os.path.join(save_dir, "curves.npz"),
            losses=np.array(losses),
            accs=np.array(accs),
            change_accs=np.array(change_accs),
        )

        # Save config
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        # Evaluate
        print("\nEvaluating world model (autoregressive rollout)...")
        evaluate_world_model(model, params, json_str, level_i=args.level, save_dir=save_dir, history=args.history)

        # Render post-training comparison GIFs
        print("\nRendering comparison GIFs...")
        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, args.game)
        render_post_training_gifs(
            model, params, json_str, backend_render,
            search_data=dataset, level_i=args.level,
            save_dir=save_dir,
        )

    # Object names for activation visualization
    obj_names = env._canonical_ids

    # Modes that need the renderer
    need_renderer = args.play or args.render_gif
    if need_renderer:
        # May already exist from post-training GIF rendering; create if not
        try:
            backend_render
        except NameError:
            backend_render = CppPuzzleScriptBackend()
            backend_render.compile_game(ps_parser, args.game)

    if args.play:
        play_dir = os.path.join(save_dir, "play")
        play_world_model(
            model, params, json_str, backend_render,
            level_i=args.level, save_dir=play_dir,
        )
    elif args.render_gif:
        gif_path = os.path.join(save_dir, "rollout_comparison.gif")
        render_rollout_comparison(
            model, params, json_str, backend_render,
            level_i=args.level, save_path=gif_path,
        )

    print("Done!")


if __name__ == "__main__":
    main()
