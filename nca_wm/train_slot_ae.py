"""Train a slot-based game-token autoencoder.

Companion to train_token_ae.py — but here the encoder is `RuleSlotEncoder`
(from rule_attn_model.py, the architecture used for the cosine_v2 19-game
and gallery world models) and the decoder is `SlotTokenDecoder` (cross-
attention to the K-slot game latent).

Two modes:
  --init_from <ckpt_dir>   : initialize encoder params from a trained
        RuleAttnNCAWorldModel checkpoint.
  (default)                : fresh random encoder.

The slot encoder produces (N_games, K, d_slot) — much richer than the
single-z FiLM encoder. The decoder uses cross-attention at every layer to
attend into those K slots, mirroring the per-step cross-attention used in
the world model.

Usage:
    python -m nca_wm.train_slot_ae \\
        --init_from nca_wm/logs/multi_gallery_..._rule_attn_..._s-0 \\
        --freeze_encoder \\
        --save_dir nca_wm/logs/slot_ae_gallery
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.core import unfreeze

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.rule_attn_model import RuleSlotEncoder, VectorQuantizer
from nca_wm.token_decoder import SlotTokenDecoder, decoder_loss, shift_right
from nca_wm.tokenize_game import VOCAB_SIZE_BASE


def _load_game_infos(ckpt_dir: str):
    infos_path = os.path.join(ckpt_dir, "game_infos.pkl")
    if not os.path.isfile(infos_path):
        raise FileNotFoundError(f"No game_infos.pkl at {infos_path}")
    with open(infos_path, "rb") as f:
        return pickle.load(f)


def _tokens_to_arrays(game_infos, max_len):
    """Pad each game's token_ids to (N_games, max_len) int32 + bool mask."""
    N = len(game_infos)
    out = np.zeros((N, max_len), dtype=np.int32)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, info in enumerate(game_infos):
        tids = info.get("token_ids", [])
        L = min(len(tids), max_len)
        if L > 0:
            out[i, :L] = tids[:L]
            mask[i, :L] = True
    return out, mask


def _load_encoder_params(ckpt_dir: str):
    """Extract the `game_encoder` subtree from a trained RuleAttn world model."""
    params_path = os.path.join(ckpt_dir, "params.pkl")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"No params.pkl at {params_path}")
    with open(params_path, "rb") as f:
        full_params = pickle.load(f)
    if isinstance(full_params, dict) and "wm" in full_params:
        full_params = full_params["wm"]
    enc_params = full_params.get("params", full_params)
    for key in ("game_encoder", "encoder"):
        if key in enc_params:
            return {"params": enc_params[key]}
    raise KeyError(
        f"No game_encoder subtree in {params_path}; "
        f"top-level keys: {list(enc_params.keys())}"
    )


def _apply_slot_pre_norm(slots, mode: str, eps: float = 1e-6):
    """Optional parameter-free normalization before vector quantization."""
    if mode == "none":
        return slots
    if mode == "layernorm":
        mean = jnp.mean(slots, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(slots - mean), axis=-1, keepdims=True)
        return (slots - mean) * jax.lax.rsqrt(var + eps)
    if mode == "l2":
        norm = jnp.linalg.norm(slots, axis=-1, keepdims=True)
        return slots / jnp.maximum(norm, eps) * jnp.sqrt(float(slots.shape[-1]))
    raise ValueError(f"Unknown slot_pre_norm={mode}")


def _apply_slot_pre_norm_np(slots: np.ndarray, mode: str, eps: float = 1e-6):
    """Numpy equivalent used for VQ codebook initialization diagnostics."""
    if mode == "none":
        return slots
    if mode == "layernorm":
        mean = slots.mean(axis=-1, keepdims=True)
        var = np.square(slots - mean).mean(axis=-1, keepdims=True)
        return (slots - mean) / np.sqrt(var + eps)
    if mode == "l2":
        norm = np.linalg.norm(slots, axis=-1, keepdims=True)
        return slots / np.maximum(norm, eps) * np.sqrt(float(slots.shape[-1]))
    raise ValueError(f"Unknown slot_pre_norm={mode}")


def _sample_rows_with_fill(flat: np.ndarray, n_rows: int, rng: np.random.Generator):
    """Sample data rows, then fill any extra codebook entries with jitter."""
    n_data, d = flat.shape
    if n_data == 0:
        raise ValueError("Cannot initialize VQ codebook from an empty slot set")
    n_base = min(n_rows, n_data)
    base_idx = rng.choice(n_data, size=n_base, replace=False)
    rows = flat[base_idx].astype(np.float32, copy=True)
    if n_base < n_rows:
        fill_idx = rng.choice(n_data, size=n_rows - n_base, replace=True)
        scale = max(float(flat.std()), 1e-3) * 1e-3
        filler = flat[fill_idx] + rng.normal(0.0, scale, size=(n_rows - n_base, d))
        rows = np.concatenate([rows, filler.astype(np.float32)], axis=0)
    rng.shuffle(rows, axis=0)
    return rows


def _kmeans_codebook(flat: np.ndarray, n_rows: int, rng: np.random.Generator,
                     n_iter: int = 25):
    """Small numpy k-means initializer; caps clusters at available slots."""
    n_data, d = flat.shape
    k = min(n_rows, n_data)
    centers = _sample_rows_with_fill(flat, k, rng)
    for _ in range(n_iter):
        dists = (
            np.square(flat).sum(axis=1, keepdims=True)
            - 2.0 * flat @ centers.T
            + np.square(centers).sum(axis=1, keepdims=True).T
        )
        labels = dists.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            assigned = flat[labels == i]
            if len(assigned):
                new_centers[i] = assigned.mean(axis=0)
        if np.allclose(new_centers, centers, rtol=1e-5, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    if k < n_rows:
        fill_idx = rng.choice(n_data, size=n_rows - k, replace=True)
        scale = max(float(flat.std()), 1e-3) * 1e-3
        filler = flat[fill_idx] + rng.normal(0.0, scale, size=(n_rows - k, d))
        centers = np.concatenate([centers, filler.astype(np.float32)], axis=0)
    rng.shuffle(centers, axis=0)
    return centers.astype(np.float32)


def _vq_code_histogram_np(indices: np.ndarray, codebook_size: int):
    counts = np.bincount(indices.reshape(-1), minlength=codebook_size)
    return {str(int(i)): int(c) for i, c in enumerate(counts) if c > 0}


def _vq_joint_histogram_np(indices: np.ndarray, codebook_size: int):
    """Histogram joint residual-code tuples from (Q, ...slot...) indices."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim < 2:
        return _vq_code_histogram_np(idx, codebook_size)
    flat = idx.reshape(idx.shape[0], -1).T
    counts = {}
    for row in flat:
        key = ",".join(str(int(x)) for x in row)
        counts[key] = counts.get(key, 0) + 1
    return {k: int(v) for k, v in sorted(counts.items())}


def _vq_pairwise_stats(codebook, slots, codebook_size: int,
                       entropy_temp: float, margin_target: float,
                       balance_target: int, balance_temp: float):
    """Shared nearest-code diagnostics and optional losses for one codebook."""
    flat = slots.reshape(-1, slots.shape[-1])
    x_sq = jnp.sum(jnp.square(flat), axis=-1, keepdims=True)
    c_sq = jnp.sum(jnp.square(codebook), axis=-1)[None, :]
    dists = x_sq - 2.0 * (flat @ codebook.T) + c_sq
    idx_flat = jnp.argmin(dists, axis=-1)

    temp = jnp.maximum(jnp.asarray(entropy_temp, dtype=slots.dtype), 1e-6)
    usage_soft = jax.nn.softmax(-dists / temp, axis=-1)
    avg_probs = jnp.mean(usage_soft, axis=0)
    entropy = -jnp.sum(avg_probs * jnp.log(jnp.clip(avg_probs, 1e-9, 1.0)))
    usage_loss = jnp.log(float(codebook_size)) - entropy
    soft_perplexity = jnp.exp(entropy)

    nearest_two = jnp.sort(dists, axis=-1)[:, :2]
    margins = nearest_two[:, 1] - nearest_two[:, 0]
    target = jnp.asarray(margin_target, dtype=slots.dtype)
    margin_loss = jnp.mean(jax.nn.relu(target - margins))
    mean_margin = jnp.mean(margins)

    hard_probs = jax.nn.one_hot(idx_flat, codebook_size, dtype=slots.dtype)
    bal_temp = jnp.maximum(jnp.asarray(balance_temp, dtype=slots.dtype), 1e-6)
    soft_probs = jax.nn.softmax(-dists / bal_temp, axis=-1)
    st_probs = hard_probs + soft_probs - jax.lax.stop_gradient(soft_probs)
    avg_hard_probs = jnp.mean(st_probs, axis=0)
    balance_entropy = -jnp.sum(
        avg_hard_probs * jnp.log(jnp.clip(avg_hard_probs, 1e-9, 1.0))
    )
    max_target = min(int(balance_target), int(flat.shape[0]), int(codebook_size))
    target_entropy = jnp.log(jnp.asarray(max(max_target, 1), dtype=slots.dtype))
    balance_loss = jax.nn.relu(target_entropy - balance_entropy)
    balance_perplexity = jnp.exp(balance_entropy)
    hard_util = jnp.count_nonzero(jnp.bincount(idx_flat, length=codebook_size))

    return {
        "dists": dists,
        "idx_flat": idx_flat,
        "usage_loss": usage_loss,
        "soft_perplexity": soft_perplexity,
        "margin_loss": margin_loss,
        "mean_margin": mean_margin,
        "balance_loss": balance_loss,
        "balance_perplexity": balance_perplexity,
        "hard_util": hard_util,
    }


def _vq_assignment_losses(latent_p, slots, margin_target: float,
                          balance_target: int, balance_temp: float):
    """Diagnostics/losses for fragile and imbalanced hard VQ assignments.

    The balance term uses straight-through hard one-hot assignments: the
    forward value is based on hard counts, while gradients flow through soft
    distances. This keeps the diagnostic aligned with saved hard code use.
    """
    codebook = jnp.asarray(latent_p["params"]["codebook"])
    stats = _vq_pairwise_stats(
        codebook, slots, int(codebook.shape[0]), 1.0, margin_target,
        balance_target, balance_temp,
    )
    return (
        stats["margin_loss"], stats["mean_margin"],
        stats["balance_loss"], stats["balance_perplexity"],
    )


def _vq_quantize(latent_p, slots, codebook_size: int, entropy_temp: float,
                 assign_mode: str, assign_temp, rng_key, deterministic: bool):
    """VQ quantization with optional soft/straight-through soft assignment."""
    codebook = jnp.asarray(latent_p["params"]["codebook"])
    flat = slots.reshape(-1, slots.shape[-1])
    x_sq = jnp.sum(jnp.square(flat), axis=-1, keepdims=True)
    c_sq = jnp.sum(jnp.square(codebook), axis=-1)[None, :]
    dists = x_sq - 2.0 * (flat @ codebook.T) + c_sq
    idx_flat = jnp.argmin(dists, axis=-1)
    hard_probs = jax.nn.one_hot(idx_flat, codebook_size, dtype=slots.dtype)

    assign_temp = jnp.maximum(jnp.asarray(assign_temp, dtype=slots.dtype), 1e-6)
    assign_logits = -dists
    if assign_mode == "gumbel_st" and not deterministic:
        assign_logits = assign_logits + jax.random.gumbel(
            rng_key, assign_logits.shape, dtype=assign_logits.dtype
        )
    assign_soft = jax.nn.softmax(assign_logits / assign_temp, axis=-1)
    if assign_mode == "hard":
        assign = hard_probs
    elif assign_mode == "soft":
        assign = assign_soft
    elif assign_mode in ("soft_st", "gumbel_st"):
        if assign_mode == "gumbel_st" and not deterministic:
            sample_idx = jnp.argmax(assign_logits, axis=-1)
            sample_hard = jax.nn.one_hot(
                sample_idx, codebook_size, dtype=slots.dtype
            )
        else:
            sample_hard = hard_probs
        assign = sample_hard + assign_soft - jax.lax.stop_gradient(assign_soft)
    else:
        raise ValueError(f"Unknown vq_assign_mode={assign_mode}")

    quantized_flat = assign @ codebook
    quantized = quantized_flat.reshape(slots.shape)
    indices = idx_flat.reshape(slots.shape[:-1])

    temp = jnp.maximum(jnp.asarray(entropy_temp, dtype=slots.dtype), 1e-6)
    usage_soft = jax.nn.softmax(-dists / temp, axis=-1)
    avg_probs = jnp.mean(usage_soft, axis=0)
    entropy = -jnp.sum(avg_probs * jnp.log(jnp.clip(avg_probs, 1e-9, 1.0)))
    usage_loss = jnp.log(float(codebook_size)) - entropy
    soft_perplexity = jnp.exp(entropy)

    assign_avg_probs = jnp.mean(assign_soft, axis=0)
    assign_entropy = -jnp.sum(
        assign_avg_probs * jnp.log(jnp.clip(assign_avg_probs, 1e-9, 1.0))
    )
    assign_perplexity = jnp.exp(assign_entropy)

    codebook_loss = jnp.mean(
        (jax.lax.stop_gradient(slots) - quantized) ** 2
    )
    commitment_loss = jnp.mean(
        (slots - jax.lax.stop_gradient(quantized)) ** 2
    )
    if assign_mode == "soft":
        slots_q = quantized
    else:
        slots_q = slots + jax.lax.stop_gradient(quantized - slots)
    return (
        slots_q, codebook_loss, commitment_loss, indices,
        usage_loss, soft_perplexity, assign_perplexity,
    )


def _vq_quantize_codebook(codebook, slots, entropy_temp: float,
                          assign_mode: str, assign_temp, rng_key,
                          deterministic: bool):
    """Quantize slots against one explicit codebook array."""
    codebook = jnp.asarray(codebook)
    codebook_size = int(codebook.shape[0])
    flat = slots.reshape(-1, slots.shape[-1])
    stats = _vq_pairwise_stats(
        codebook, slots, codebook_size, entropy_temp,
        margin_target=0.0, balance_target=codebook_size, balance_temp=1.0,
    )
    dists = stats["dists"]
    idx_flat = stats["idx_flat"]
    hard_probs = jax.nn.one_hot(idx_flat, codebook_size, dtype=slots.dtype)

    assign_temp = jnp.maximum(jnp.asarray(assign_temp, dtype=slots.dtype), 1e-6)
    assign_logits = -dists
    if assign_mode == "gumbel_st" and not deterministic:
        assign_logits = assign_logits + jax.random.gumbel(
            rng_key, assign_logits.shape, dtype=assign_logits.dtype
        )
    assign_soft = jax.nn.softmax(assign_logits / assign_temp, axis=-1)
    if assign_mode == "hard":
        assign = hard_probs
    elif assign_mode == "soft":
        assign = assign_soft
    elif assign_mode in ("soft_st", "gumbel_st"):
        if assign_mode == "gumbel_st" and not deterministic:
            sample_idx = jnp.argmax(assign_logits, axis=-1)
            sample_hard = jax.nn.one_hot(
                sample_idx, codebook_size, dtype=slots.dtype
            )
        else:
            sample_hard = hard_probs
        assign = sample_hard + assign_soft - jax.lax.stop_gradient(assign_soft)
    else:
        raise ValueError(f"Unknown vq_assign_mode={assign_mode}")

    quantized_flat = assign @ codebook
    quantized = quantized_flat.reshape(slots.shape)
    indices = idx_flat.reshape(slots.shape[:-1])
    assign_avg_probs = jnp.mean(assign_soft, axis=0)
    assign_entropy = -jnp.sum(
        assign_avg_probs * jnp.log(jnp.clip(assign_avg_probs, 1e-9, 1.0))
    )
    assign_perplexity = jnp.exp(assign_entropy)
    codebook_loss = jnp.mean(
        (jax.lax.stop_gradient(slots) - quantized) ** 2
    )
    commitment_loss = jnp.mean(
        (slots - jax.lax.stop_gradient(quantized)) ** 2
    )
    if assign_mode == "soft":
        slots_q = quantized
    else:
        slots_q = slots + jax.lax.stop_gradient(quantized - slots)
    return (
        slots_q, quantized, codebook_loss, commitment_loss, indices,
        stats["usage_loss"], stats["soft_perplexity"], assign_perplexity,
    )


def _residual_vq_quantize(latent_p, slots, entropy_temp: float,
                          assign_mode: str, assign_temp, rng_key,
                          deterministic: bool, margin_target: float,
                          balance_target: int, balance_temp: float):
    """Residual VQ over a stack of codebooks with per-stage diagnostics."""
    codebooks = jnp.asarray(latent_p["params"]["codebook"])
    residual = slots
    quantized_sum = jnp.zeros_like(slots)
    cb_losses = []
    commit_losses = []
    usage_losses = []
    soft_perps = []
    assign_perps = []
    margin_losses = []
    balance_losses = []
    balance_perps = []
    indices = []
    stage_utils = []
    mean_margins = []
    residual_norms = [jnp.mean(jnp.linalg.norm(residual, axis=-1))]
    for q in range(int(codebooks.shape[0])):
        stage_key = jax.random.fold_in(rng_key, q)
        (stage_q, quantized, cb, commit, idx, usage, perp,
         assign_perp) = _vq_quantize_codebook(
            codebooks[q], residual, entropy_temp, assign_mode, assign_temp,
            stage_key, deterministic,
        )
        quantized_sum = quantized_sum + stage_q
        residual = residual - jax.lax.stop_gradient(quantized)
        stage_stats = _vq_pairwise_stats(
            codebooks[q], residual + jax.lax.stop_gradient(quantized),
            int(codebooks.shape[1]), entropy_temp, margin_target,
            balance_target, balance_temp,
        )
        cb_losses.append(cb)
        commit_losses.append(commit)
        usage_losses.append(usage)
        soft_perps.append(perp)
        assign_perps.append(assign_perp)
        margin_losses.append(stage_stats["margin_loss"])
        balance_losses.append(stage_stats["balance_loss"])
        balance_perps.append(stage_stats["balance_perplexity"])
        indices.append(idx)
        stage_utils.append(stage_stats["hard_util"])
        mean_margins.append(stage_stats["mean_margin"])
        residual_norms.append(jnp.mean(jnp.linalg.norm(residual, axis=-1)))

    indices = jnp.stack(indices, axis=0)
    stage_utils = jnp.asarray(stage_utils, dtype=slots.dtype)
    # Count unique residual-code tuples among slots. The number of slot rows is
    # static and small, so bounded unique works under jit.
    joint_rows = jnp.moveaxis(indices, 0, -1).reshape(-1, indices.shape[0])
    multipliers = (
        jnp.asarray(int(codebooks.shape[1]), dtype=jnp.int32)
        ** jnp.arange(indices.shape[0], dtype=jnp.int32)
    )
    joint_codes = jnp.sum(joint_rows.astype(jnp.int32) * multipliers[None, :], axis=-1)
    unique_codes = jnp.unique(joint_codes, size=joint_codes.shape[0], fill_value=-1)
    joint_util = jnp.count_nonzero(unique_codes >= 0)
    quantized_st = slots + jax.lax.stop_gradient(quantized_sum - slots)
    return (
        quantized_st,
        jnp.mean(jnp.asarray(cb_losses)),
        jnp.mean(jnp.asarray(commit_losses)),
        indices,
        jnp.mean(jnp.asarray(usage_losses)),
        jnp.mean(jnp.asarray(soft_perps)),
        jnp.mean(jnp.asarray(assign_perps)),
        jnp.mean(jnp.asarray(margin_losses)),
        jnp.mean(jnp.asarray(balance_losses)),
        jnp.mean(jnp.asarray(balance_perps)),
        stage_utils,
        joint_util.astype(slots.dtype),
        jnp.asarray(mean_margins),
        jnp.asarray(residual_norms),
    )


def _joint_util_from_indices(indices, codebook_size: int):
    joint_rows = jnp.moveaxis(indices, 0, -1).reshape(-1, indices.shape[0])
    multipliers = (
        jnp.asarray(int(codebook_size), dtype=jnp.int32)
        ** jnp.arange(indices.shape[0], dtype=jnp.int32)
    )
    joint_codes = jnp.sum(joint_rows.astype(jnp.int32) * multipliers[None, :], axis=-1)
    unique_codes = jnp.unique(joint_codes, size=joint_codes.shape[0], fill_value=-1)
    return jnp.count_nonzero(unique_codes >= 0)


def _product_vq_quantize(latent_p, slots, entropy_temp: float,
                         assign_mode: str, assign_temp, rng_key,
                         deterministic: bool, margin_target: float,
                         balance_target: int, balance_temp: float):
    """Product quantization over fixed contiguous slot subspaces."""
    codebooks = jnp.asarray(latent_p["params"]["codebook"])
    chunks = jnp.split(slots, int(codebooks.shape[0]), axis=-1)
    quantized_chunks = []
    cb_losses = []
    commit_losses = []
    usage_losses = []
    soft_perps = []
    assign_perps = []
    margin_losses = []
    balance_losses = []
    balance_perps = []
    indices = []
    stage_utils = []
    mean_margins = []
    chunk_norms = []
    for q, chunk in enumerate(chunks):
        stage_key = jax.random.fold_in(rng_key, q)
        (stage_q, _, cb, commit, idx, usage, perp,
         assign_perp) = _vq_quantize_codebook(
            codebooks[q], chunk, entropy_temp, assign_mode, assign_temp,
            stage_key, deterministic,
        )
        stage_stats = _vq_pairwise_stats(
            codebooks[q], chunk, int(codebooks.shape[1]), entropy_temp,
            margin_target, balance_target, balance_temp,
        )
        quantized_chunks.append(stage_q)
        cb_losses.append(cb)
        commit_losses.append(commit)
        usage_losses.append(usage)
        soft_perps.append(perp)
        assign_perps.append(assign_perp)
        margin_losses.append(stage_stats["margin_loss"])
        balance_losses.append(stage_stats["balance_loss"])
        balance_perps.append(stage_stats["balance_perplexity"])
        indices.append(idx)
        stage_utils.append(stage_stats["hard_util"])
        mean_margins.append(stage_stats["mean_margin"])
        chunk_norms.append(jnp.mean(jnp.linalg.norm(chunk, axis=-1)))

    indices = jnp.stack(indices, axis=0)
    quantized = jnp.concatenate(quantized_chunks, axis=-1)
    return (
        quantized,
        jnp.mean(jnp.asarray(cb_losses)),
        jnp.mean(jnp.asarray(commit_losses)),
        indices,
        jnp.mean(jnp.asarray(usage_losses)),
        jnp.mean(jnp.asarray(soft_perps)),
        jnp.mean(jnp.asarray(assign_perps)),
        jnp.mean(jnp.asarray(margin_losses)),
        jnp.mean(jnp.asarray(balance_losses)),
        jnp.mean(jnp.asarray(balance_perps)),
        jnp.asarray(stage_utils, dtype=slots.dtype),
        _joint_util_from_indices(indices, int(codebooks.shape[1])).astype(slots.dtype),
        jnp.asarray(mean_margins),
        jnp.asarray(chunk_norms),
    )


class VQCodebookStack(nn.Module):
    """Parameter holder for one or more VQ codebooks."""
    n_codebooks: int = 1
    codebook_size: int = 1024
    d_slot: int = 64

    @nn.compact
    def __call__(self, slots):
        return self.param(
            "codebook",
            nn.initializers.normal(stddev=1.0 / (self.d_slot ** 0.5)),
            (self.n_codebooks, self.codebook_size, self.d_slot),
        )


def _annealed_value(step: int, start: float, end: float, n_steps: int):
    if n_steps <= 0:
        return float(end)
    frac = min(max(float(step) / float(n_steps), 0.0), 1.0)
    if start > 0.0 and end > 0.0:
        return float(np.exp(np.log(start) * (1.0 - frac) + np.log(end) * frac))
    return float(start * (1.0 - frac) + end * frac)


def _tree_to_python(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


class SlotGaussianBottleneck(nn.Module):
    """Per-slot Gaussian posterior q(z|slots) with a standard-normal prior."""
    d_slot: int = 64
    logvar_min: float = -10.0
    logvar_max: float = 5.0

    @nn.compact
    def __call__(self, slots, rng=None, deterministic: bool = False):
        # Residual mean starts as the identity map, so VAE training begins near
        # the deterministic AE path instead of destroying a pretrained encoder.
        mu_delta = nn.Dense(
            self.d_slot,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="mu_delta",
        )(slots)
        mu = slots + mu_delta
        logvar = nn.Dense(
            self.d_slot,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-6.0),
            name="logvar",
        )(slots)
        logvar = jnp.clip(logvar, self.logvar_min, self.logvar_max)
        if deterministic:
            z = mu
        else:
            if rng is None:
                raise ValueError("SlotGaussianBottleneck needs rng when sampling")
            eps = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
            z = mu + eps * jnp.exp(0.5 * logvar)

        # Standard VAE KL, reported both per example and per latent dim.
        kl_per_example = 0.5 * jnp.sum(
            jnp.square(mu) + jnp.exp(logvar) - 1.0 - logvar,
            axis=(1, 2),
        )
        kl_total = jnp.mean(kl_per_example)
        kl_per_dim = kl_total / float(slots.shape[1] * slots.shape[2])
        sigma_mean = jnp.mean(jnp.exp(0.5 * logvar))
        return z, kl_total, kl_per_dim, sigma_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", type=str, default=None,
                   help="Trained rule-attn world-model dir.")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder params; train only decoder.")
    p.add_argument("--vocab_size", type=int, default=None,
                   help="Token vocab size. Defaults to checkpoint config "
                        "`vocab_size`, or max token id + 1, falling back to "
                        f"VOCAB_SIZE_BASE={VOCAB_SIZE_BASE}.")
    p.add_argument("--max_seq_len", type=int, default=192)
    # Encoder shape — must match the world model's encoder if --init_from
    p.add_argument("--enc_d_model", type=int, default=64)
    p.add_argument("--enc_n_self_layers", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=16)
    p.add_argument("--d_slot", type=int, default=64)
    p.add_argument("--enc_n_heads", type=int, default=4)
    # Decoder shape (independent — newly trained)
    p.add_argument("--dec_d_model", type=int, default=128)
    p.add_argument("--dec_n_layers", type=int, default=4)
    p.add_argument("--dec_n_heads", type=int, default=4)
    # Latent bottleneck variant.
    p.add_argument("--latent_model", choices=["ae", "vae", "vqvae"],
                   default="ae",
                   help="'ae' = deterministic slots; 'vae' = Gaussian slots "
                        "with KL; 'vqvae' = quantized slots with VQ losses.")
    p.add_argument("--vae_kl_weight", type=float, default=1e-4,
                   help="Multiplier on total KL nats/example for --latent_model vae.")
    p.add_argument("--vq_codebook_size", type=int, default=1024)
    p.add_argument("--vq_quantizer", choices=["single", "residual", "product"],
                   default="single",
                   help="VQ structure. 'single' preserves the existing one "
                        "codebook path; 'residual' applies multiple codebooks "
                        "sequentially to slot residuals; 'product' quantizes "
                        "fixed slot subspaces independently.")
    p.add_argument("--vq_num_quantizers", type=int, default=1,
                   help="Number of codebooks for --vq_quantizer residual.")
    p.add_argument("--vq_residual_codebook_size", type=int, default=256,
                   help="Per-stage codebook size for residual VQ.")
    p.add_argument("--vq_commitment_weight", type=float, default=0.25)
    p.add_argument("--vq_loss_weight", type=float, default=1.0)
    p.add_argument("--vq_usage_loss_weight", type=float, default=0.0)
    p.add_argument("--vq_entropy_temp", type=float, default=1.0)
    p.add_argument("--vq_margin_loss_weight", type=float, default=0.0,
                   help="Optional hard-assignment margin loss weight. The loss "
                        "penalizes nearest/second-nearest squared-distance "
                        "gaps below --vq_margin_target.")
    p.add_argument("--vq_margin_target", type=float, default=0.05,
                   help="Target squared-distance gap for the optional VQ "
                        "assignment margin loss.")
    p.add_argument("--vq_hard_balance_loss_weight", type=float, default=0.0,
                   help="Optional straight-through hard-assignment balance "
                        "loss weight. Encourages hard assignment perplexity "
                        "to reach --vq_hard_balance_target.")
    p.add_argument("--vq_hard_balance_target", type=int, default=64,
                   help="Target number of active hard VQ codes for the "
                        "optional hard-assignment balance loss.")
    p.add_argument("--vq_hard_balance_temp", type=float, default=1.0,
                   help="Softmax temperature for gradients of the optional "
                        "straight-through hard balance loss.")
    p.add_argument("--vq_assign_mode", choices=["hard", "soft", "soft_st", "gumbel_st"],
                   default="hard",
                   help="Assignment used by VQ-VAE token AE. 'hard' preserves "
                        "the original argmin VQ path; 'soft' uses annealed "
                        "soft assignments; 'soft_st' uses hard forward values "
                        "with soft assignment gradients; 'gumbel_st' samples "
                        "hard codes during training with Gumbel-softmax "
                        "gradients.")
    p.add_argument("--vq_assign_temp_start", type=float, default=1.0,
                   help="Initial temperature for soft/soft_st VQ assignment.")
    p.add_argument("--vq_assign_temp_end", type=float, default=0.05,
                   help="Final temperature for soft/soft_st VQ assignment.")
    p.add_argument("--vq_assign_anneal_steps", type=int, default=5000,
                   help="Number of updates over which to anneal the VQ "
                        "assignment temperature.")
    p.add_argument("--vq_init", choices=["random", "slot_sample", "kmeans"],
                   default="random",
                   help="VQ codebook initializer. Non-random modes initialize "
                        "from the encoder slots after optional slot_pre_norm.")
    p.add_argument("--slot_pre_norm", choices=["none", "layernorm", "l2"],
                   default="none",
                   help="Parameter-free normalization applied before VQ only.")
    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_updates", type=int, default=20000)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--game_infos", type=str, default=None,
                   help="Path to game_infos.pkl (defaults to --init_from dir).")
    args = p.parse_args()

    if args.latent_model != "vqvae" and args.vq_quantizer != "single":
        p.error("--vq_quantizer is only meaningful with --latent_model vqvae")
    if args.vq_quantizer == "single":
        args.vq_num_quantizers = 1
    if args.vq_num_quantizers < 1:
        p.error("--vq_num_quantizers must be >= 1")

    os.makedirs(args.save_dir, exist_ok=True)

    infos_dir = args.game_infos or args.init_from
    if infos_dir is None:
        raise ValueError("Need --init_from or --game_infos for game_infos.pkl")
    if os.path.isdir(infos_dir):
        game_infos = _load_game_infos(infos_dir)
    else:
        with open(infos_dir, "rb") as f:
            game_infos = pickle.load(f)
    N_games = len(game_infos)
    print(f"Loaded {N_games} games from {infos_dir}")

    if args.vocab_size is None:
        cfg_vocab = None
        if args.init_from:
            cfg_path = os.path.join(args.init_from, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg_vocab = json.load(f).get("vocab_size")
        max_token = max(
            (max(info.get("token_ids", [0])) for info in game_infos),
            default=VOCAB_SIZE_BASE - 1,
        )
        args.vocab_size = int(cfg_vocab or (max_token + 1) or VOCAB_SIZE_BASE)
    print(f"Using vocab_size={args.vocab_size}")

    # Match train.py's encoder shape: max_seq_len = max_tok_len + 1.
    max_tok_len = max(len(info.get("token_ids", [])) for info in game_infos)
    enc_max_seq_len = max(max_tok_len + 1, 2)
    print(f"Encoder max_seq_len = {enc_max_seq_len} (max token sequence + 1)")

    tokens_np, mask_np = _tokens_to_arrays(game_infos, enc_max_seq_len)
    tok_lens = mask_np.sum(axis=1)
    print(f"Token lengths: min={tok_lens.min()}, max={tok_lens.max()}, "
          f"mean={tok_lens.mean():.1f}")

    # Models
    encoder = RuleSlotEncoder(
        vocab_size=args.vocab_size + 1,
        max_seq_len=enc_max_seq_len,
        d_model=args.enc_d_model,
        n_self_layers=args.enc_n_self_layers,
        n_slots=args.n_slots,
        d_slot=args.d_slot,
        n_heads=args.enc_n_heads,
    )
    decoder = SlotTokenDecoder(
        vocab_size=args.vocab_size,
        max_seq_len=enc_max_seq_len,
        d_model=args.dec_d_model,
        n_layers=args.dec_n_layers,
        n_heads=args.dec_n_heads,
        d_slot=args.d_slot,
    )
    vae_bottleneck = None
    vq_bottleneck = None
    vq_codebook_size = args.vq_codebook_size
    if args.latent_model == "vae":
        vae_bottleneck = SlotGaussianBottleneck(d_slot=args.d_slot)
    elif args.latent_model == "vqvae":
        if args.vq_quantizer == "single":
            vq_bottleneck = VectorQuantizer(
                codebook_size=args.vq_codebook_size,
                d_slot=args.d_slot,
                commitment_weight=args.vq_commitment_weight,
                entropy_temp=args.vq_entropy_temp,
            )
            vq_codebook_size = args.vq_codebook_size
        elif args.vq_quantizer in ("residual", "product"):
            codebook_dim = args.d_slot
            if args.vq_quantizer == "product":
                if args.d_slot % args.vq_num_quantizers != 0:
                    p.error("--d_slot must be divisible by --vq_num_quantizers for product VQ")
                codebook_dim = args.d_slot // args.vq_num_quantizers
            vq_bottleneck = VQCodebookStack(
                n_codebooks=args.vq_num_quantizers,
                codebook_size=args.vq_residual_codebook_size,
                d_slot=codebook_dim,
            )
            vq_codebook_size = args.vq_residual_codebook_size
        else:
            raise ValueError(f"Unknown vq_quantizer={args.vq_quantizer}")

    rng = jax.random.PRNGKey(args.seed)
    rng, enc_rng, dec_rng = jax.random.split(rng, 3)
    toks_j = jnp.array(tokens_np[:1])
    mask_j = jnp.array(mask_np[:1])
    enc_params = encoder.init(enc_rng, toks_j, mask_j, deterministic=True)
    slots_dummy = encoder.apply(enc_params, toks_j, mask_j, deterministic=True)
    dec_params = decoder.init(dec_rng, toks_j, slots_dummy, deterministic=True)
    latent_params = None
    if vae_bottleneck is not None:
        rng, lat_rng, sample_rng = jax.random.split(rng, 3)
        latent_params = vae_bottleneck.init(
            lat_rng, slots_dummy, sample_rng, deterministic=False
        )
    elif vq_bottleneck is not None:
        rng, lat_rng = jax.random.split(rng)
        latent_params = vq_bottleneck.init(lat_rng, slots_dummy)

    if args.init_from:
        loaded = _load_encoder_params(args.init_from)
        try:
            _ = encoder.apply(loaded, toks_j, mask_j, deterministic=True)
            enc_params = loaded
            print(f"Loaded encoder from {args.init_from}")
        except Exception as e:
            print(f"WARNING: encoder shape mismatch ({e}); using random init.")

    if vq_bottleneck is not None and args.vq_init != "random":
        init_slots = encoder.apply(
            enc_params, jnp.array(tokens_np), jnp.array(mask_np),
            deterministic=True,
        )
        init_slots_np = np.array(_apply_slot_pre_norm(
            init_slots, args.slot_pre_norm
        )).reshape(-1, args.d_slot)
        np_rng = np.random.default_rng(args.seed)
        codebooks_np = []
        if args.vq_quantizer == "product":
            for chunk_np in np.split(init_slots_np, args.vq_num_quantizers, axis=1):
                if args.vq_init == "slot_sample":
                    cb_np = _sample_rows_with_fill(
                        chunk_np, vq_codebook_size, np_rng
                    )
                elif args.vq_init == "kmeans":
                    cb_np = _kmeans_codebook(
                        chunk_np, vq_codebook_size, np_rng
                    )
                else:
                    raise ValueError(f"Unknown vq_init={args.vq_init}")
                codebooks_np.append(cb_np)
        else:
            residual_np = init_slots_np.copy()
            for _ in range(args.vq_num_quantizers):
                if args.vq_init == "slot_sample":
                    cb_np = _sample_rows_with_fill(
                        residual_np, vq_codebook_size, np_rng
                    )
                elif args.vq_init == "kmeans":
                    cb_np = _kmeans_codebook(
                        residual_np, vq_codebook_size, np_rng
                    )
                else:
                    raise ValueError(f"Unknown vq_init={args.vq_init}")
                codebooks_np.append(cb_np)
                if args.vq_quantizer == "residual":
                    dists_np = (
                        np.square(residual_np).sum(axis=1, keepdims=True)
                        - 2.0 * residual_np @ cb_np.T
                        + np.square(cb_np).sum(axis=1, keepdims=True).T
                    )
                    residual_np = residual_np - cb_np[dists_np.argmin(axis=1)]
        if args.vq_quantizer == "single":
            codebook_np = codebooks_np[0]
        else:
            codebook_np = np.stack(codebooks_np, axis=0)
        latent_mut = unfreeze(latent_params)
        latent_mut["params"]["codebook"] = jnp.asarray(codebook_np)
        latent_params = latent_mut
        print(
            f"Initialized VQ codebook with {args.vq_init} from "
            f"{init_slots_np.shape[0]} slots (slot_pre_norm={args.slot_pre_norm}, "
            f"vq_quantizer={args.vq_quantizer}, n_codebooks={args.vq_num_quantizers}, "
            f"codebook_size={vq_codebook_size})"
        )

    n_enc = sum(p.size for p in jax.tree_util.tree_leaves(enc_params))
    n_dec = sum(p.size for p in jax.tree_util.tree_leaves(dec_params))
    print(f"Encoder params: {n_enc:,}   Decoder params: {n_dec:,}")

    if args.freeze_encoder:
        trainable = {"dec": dec_params}
        frozen = {"enc": enc_params}
    else:
        trainable = {"enc": enc_params, "dec": dec_params}
        frozen = {}
    if latent_params is not None:
        trainable["latent"] = latent_params

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(trainable)

    def encode_bottleneck(latent_p, slots, rng_key, deterministic, assign_temp):
        z = jnp.asarray(0.0, dtype=slots.dtype)
        if args.latent_model == "ae":
            zero = jnp.asarray(0.0, dtype=slots.dtype)
            return slots, (
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                jnp.zeros((1,), dtype=slots.dtype),
                jnp.zeros((1,), dtype=slots.dtype),
                jnp.zeros((1,), dtype=slots.dtype),
            )
        if args.latent_model == "vae":
            z_slots, kl_total, kl_per_dim, sigma_mean = vae_bottleneck.apply(
                latent_p, slots, rng_key, deterministic=deterministic
            )
            return z_slots, (
                kl_total, kl_per_dim, sigma_mean, z, z, z, z, z, z, z, z,
                jnp.zeros((1,), dtype=slots.dtype),
                jnp.zeros((1,), dtype=slots.dtype),
                jnp.zeros((1,), dtype=slots.dtype),
            )
        slots = _apply_slot_pre_norm(slots, args.slot_pre_norm)
        if args.vq_quantizer == "residual":
            (z_slots, vq_cb, vq_commit, vq_indices, vq_usage, vq_perp,
             vq_assign_perp, vq_margin_loss, vq_balance_loss,
             vq_balance_perp, vq_stage_utils, vq_joint_util,
             vq_stage_margins, vq_residual_norms) = _residual_vq_quantize(
                latent_p, slots, args.vq_entropy_temp, args.vq_assign_mode,
                assign_temp, rng_key, deterministic, args.vq_margin_target,
                args.vq_hard_balance_target, args.vq_hard_balance_temp,
            )
            vq_util = vq_joint_util
            vq_mean_margin = jnp.mean(vq_stage_margins)
        elif args.vq_quantizer == "product":
            (z_slots, vq_cb, vq_commit, vq_indices, vq_usage, vq_perp,
             vq_assign_perp, vq_margin_loss, vq_balance_loss,
             vq_balance_perp, vq_stage_utils, vq_joint_util,
             vq_stage_margins, vq_residual_norms) = _product_vq_quantize(
                latent_p, slots, args.vq_entropy_temp, args.vq_assign_mode,
                assign_temp, rng_key, deterministic, args.vq_margin_target,
                args.vq_hard_balance_target, args.vq_hard_balance_temp,
            )
            vq_util = vq_joint_util
            vq_mean_margin = jnp.mean(vq_stage_margins)
        elif args.vq_assign_mode == "hard":
            (z_slots, vq_cb, vq_commit, vq_indices,
             vq_usage, vq_perp) = vq_bottleneck.apply(latent_p, slots)
            vq_assign_perp = vq_perp
            vq_util = jnp.count_nonzero(jnp.bincount(
                vq_indices.reshape(-1), length=args.vq_codebook_size,
            ))
            (vq_margin_loss, vq_mean_margin,
             vq_balance_loss, vq_balance_perp) = _vq_assignment_losses(
                latent_p, slots, args.vq_margin_target,
                args.vq_hard_balance_target, args.vq_hard_balance_temp,
            )
            vq_stage_utils = jnp.asarray([vq_util], dtype=slots.dtype)
            vq_stage_margins = jnp.asarray([vq_mean_margin], dtype=slots.dtype)
            vq_residual_norms = jnp.asarray([
                jnp.mean(jnp.linalg.norm(slots, axis=-1))
            ], dtype=slots.dtype)
        else:
            (z_slots, vq_cb, vq_commit, vq_indices, vq_usage,
             vq_perp, vq_assign_perp) = _vq_quantize(
                latent_p, slots, args.vq_codebook_size, args.vq_entropy_temp,
                args.vq_assign_mode, assign_temp, rng_key, deterministic,
            )
            vq_util = jnp.count_nonzero(jnp.bincount(
                vq_indices.reshape(-1), length=args.vq_codebook_size,
            ))
            (vq_margin_loss, vq_mean_margin,
             vq_balance_loss, vq_balance_perp) = _vq_assignment_losses(
                latent_p, slots, args.vq_margin_target,
                args.vq_hard_balance_target, args.vq_hard_balance_temp,
            )
            vq_stage_utils = jnp.asarray([vq_util], dtype=slots.dtype)
            vq_stage_margins = jnp.asarray([vq_mean_margin], dtype=slots.dtype)
            vq_residual_norms = jnp.asarray([
                jnp.mean(jnp.linalg.norm(slots, axis=-1))
            ], dtype=slots.dtype)
        return z_slots, (
            vq_cb, vq_commit, vq_usage, vq_perp, vq_util,
            vq_margin_loss, vq_mean_margin, vq_balance_loss, vq_balance_perp,
            jnp.asarray(assign_temp, dtype=slots.dtype), vq_assign_perp,
            vq_stage_utils, vq_stage_margins, vq_residual_norms,
        )

    def forward(enc_p, latent_p, dec_p, tokens, mask, rng_key,
                deterministic: bool, assign_temp):
        slots = encoder.apply(enc_p, tokens, mask, deterministic=True)
        latent_slots, latent_aux = encode_bottleneck(
            latent_p, slots, rng_key, deterministic, assign_temp
        )
        inputs = shift_right(tokens, bos_id=0)
        logits = decoder.apply(dec_p, inputs, latent_slots, deterministic=True)
        loss, acc = decoder_loss(logits, tokens, mask)
        return loss, (acc, logits, latent_slots, latent_aux)

    def loss_fn(trainable_params, frozen_params, tokens, mask, rng_key,
                assign_temp):
        enc_p = trainable_params.get("enc", frozen_params.get("enc"))
        latent_p = trainable_params.get("latent")
        dec_p = trainable_params["dec"]
        recon_loss, (acc, _, _, latent_aux) = forward(
            enc_p, latent_p, dec_p, tokens, mask, rng_key,
            deterministic=False, assign_temp=assign_temp,
        )
        total = recon_loss
        if args.latent_model == "vae":
            total = total + args.vae_kl_weight * latent_aux[0]
        elif args.latent_model == "vqvae":
            total = total + args.vq_loss_weight * (
                latent_aux[0] + args.vq_commitment_weight * latent_aux[1]
            )
            total = total + args.vq_usage_loss_weight * latent_aux[2]
            total = total + args.vq_margin_loss_weight * latent_aux[5]
            total = total + args.vq_hard_balance_loss_weight * latent_aux[7]
        return total, (recon_loss, acc, latent_aux)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def update(trainable_params, opt_state, frozen_params, tokens, mask,
               rng_key, assign_temp):
        (loss, aux), grads = grad_fn(
            trainable_params, frozen_params, tokens, mask, rng_key, assign_temp
        )
        updates, new_opt = optimizer.update(grads, opt_state, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        return new_params, new_opt, loss, aux

    tokens_j = jnp.array(tokens_np)
    mask_j = jnp.array(mask_np)
    history = {
        "loss": [],
        "recon_loss": [],
        "acc": [],
        "kl_total": [],
        "kl_per_dim": [],
        "sigma_mean": [],
        "vq_cb": [],
        "vq_commit": [],
        "vq_usage": [],
        "vq_perp": [],
        "vq_util": [],
        "vq_margin_loss": [],
        "vq_mean_margin": [],
        "vq_hard_balance_loss": [],
        "vq_hard_balance_perp": [],
        "vq_assign_temp": [],
        "vq_assign_perp": [],
        "vq_stage_utils": [],
        "vq_stage_margins": [],
        "vq_residual_norms": [],
    }
    t0 = time.time()
    for step in range(args.n_updates):
        rng, step_rng = jax.random.split(rng)
        assign_temp = _annealed_value(
            step, args.vq_assign_temp_start, args.vq_assign_temp_end,
            args.vq_assign_anneal_steps,
        )
        trainable, opt_state, loss, aux = update(
            trainable, opt_state, frozen, tokens_j, mask_j, step_rng,
            jnp.asarray(assign_temp, dtype=jnp.float32),
        )
        recon_loss, acc, latent_aux = aux
        history["loss"].append(float(loss))
        history["recon_loss"].append(float(recon_loss))
        history["acc"].append(float(acc))
        if args.latent_model == "vae":
            history["kl_total"].append(float(latent_aux[0]))
            history["kl_per_dim"].append(float(latent_aux[1]))
            history["sigma_mean"].append(float(latent_aux[2]))
        elif args.latent_model == "vqvae":
            history["vq_cb"].append(float(latent_aux[0]))
            history["vq_commit"].append(float(latent_aux[1]))
            history["vq_usage"].append(float(latent_aux[2]))
            history["vq_perp"].append(float(latent_aux[3]))
            history["vq_util"].append(float(latent_aux[4]))
            history["vq_margin_loss"].append(float(latent_aux[5]))
            history["vq_mean_margin"].append(float(latent_aux[6]))
            history["vq_hard_balance_loss"].append(float(latent_aux[7]))
            history["vq_hard_balance_perp"].append(float(latent_aux[8]))
            history["vq_assign_temp"].append(float(latent_aux[9]))
            history["vq_assign_perp"].append(float(latent_aux[10]))
            history["vq_stage_utils"].append(np.array(latent_aux[11]).tolist())
            history["vq_stage_margins"].append(np.array(latent_aux[12]).tolist())
            history["vq_residual_norms"].append(np.array(latent_aux[13]).tolist())
        if step % args.log_interval == 0 or step == args.n_updates - 1:
            extra = ""
            if args.latent_model == "vae":
                extra = (f"  kl={float(latent_aux[0]):.3e}"
                         f"  kl/dim={float(latent_aux[1]):.3e}"
                         f"  sigma={float(latent_aux[2]):.3f}")
            elif args.latent_model == "vqvae":
                extra = (f"  vq_util={float(latent_aux[4]):.1f}"
                         f"  vq_perp={float(latent_aux[3]):.1f}"
                         f"  vq_usage={float(latent_aux[2]):.2e}"
                         f"  vq_margin={float(latent_aux[6]):.2e}"
                         f"  vq_bal_perp={float(latent_aux[8]):.1f}"
                         f"  vq_assign_temp={float(latent_aux[9]):.3f}"
                         f"  vq_assign_perp={float(latent_aux[10]):.1f}"
                         f"  vq_stage_util={np.array(latent_aux[11]).tolist()}")
            print(f"step {step:6d}/{args.n_updates}  loss={float(loss):.4e}  "
                  f"recon={float(recon_loss):.4e}  acc={float(acc):.4f}"
                  f"{extra}  ({time.time()-t0:.0f}s)")

    enc_p = trainable.get("enc", frozen.get("enc"))
    latent_p = trainable.get("latent")
    dec_p = trainable["dec"]
    rng, eval_rng = jax.random.split(rng)
    eval_assign_temp = _annealed_value(
        args.n_updates, args.vq_assign_temp_start, args.vq_assign_temp_end,
        args.vq_assign_anneal_steps,
    )
    loss_final, (acc_final, _, _, latent_aux_final) = forward(
        enc_p, latent_p, dec_p, tokens_j, mask_j, eval_rng,
        deterministic=True, assign_temp=jnp.asarray(eval_assign_temp, dtype=jnp.float32),
    )
    print(f"\nFinal deterministic recon: loss={float(loss_final):.4e}  "
          f"acc={float(acc_final):.4f}")

    # Per-game reconstruction
    raw_slots_all = encoder.apply(enc_p, tokens_j, mask_j, deterministic=True)
    slots_all, _ = encode_bottleneck(
        latent_p, raw_slots_all, eval_rng, deterministic=True,
        assign_temp=jnp.asarray(eval_assign_temp, dtype=jnp.float32),
    )
    vq_indices_final = None
    vq_hist_final = None
    vq_stage_hist_final = None
    vq_joint_hist_final = None
    if args.latent_model == "vqvae":
        norm_slots_all = _apply_slot_pre_norm(raw_slots_all, args.slot_pre_norm)
        if args.vq_quantizer == "residual":
            (_, _, _, vq_indices_final, _, _, _, _, _, _, _, _, _, _) = (
                _residual_vq_quantize(
                    latent_p, norm_slots_all, args.vq_entropy_temp,
                    args.vq_assign_mode, jnp.asarray(eval_assign_temp, dtype=jnp.float32),
                    eval_rng, True, args.vq_margin_target,
                    args.vq_hard_balance_target, args.vq_hard_balance_temp,
                )
            )
        elif args.vq_quantizer == "product":
            (_, _, _, vq_indices_final, _, _, _, _, _, _, _, _, _, _) = (
                _product_vq_quantize(
                    latent_p, norm_slots_all, args.vq_entropy_temp,
                    args.vq_assign_mode, jnp.asarray(eval_assign_temp, dtype=jnp.float32),
                    eval_rng, True, args.vq_margin_target,
                    args.vq_hard_balance_target, args.vq_hard_balance_temp,
                )
            )
        else:
            (_, _, _, vq_indices_final, _, _) = vq_bottleneck.apply(
                latent_p, norm_slots_all
            )
        vq_indices_final_np = np.array(vq_indices_final)
        if args.vq_quantizer in ("residual", "product"):
            vq_stage_hist_final = [
                _vq_code_histogram_np(vq_indices_final_np[q], vq_codebook_size)
                for q in range(vq_indices_final_np.shape[0])
            ]
            vq_joint_hist_final = _vq_joint_histogram_np(
                vq_indices_final_np, vq_codebook_size
            )
            vq_hist_final = vq_joint_hist_final
        else:
            vq_hist_final = _vq_code_histogram_np(
                vq_indices_final_np, args.vq_codebook_size
            )
    inputs = shift_right(tokens_j, bos_id=0)
    logits_all = decoder.apply(dec_p, inputs, slots_all, deterministic=True)
    preds = jnp.argmax(logits_all, axis=-1)
    print("\nPer-game reconstruction accuracy:")
    per_game_acc = []
    for i, info in enumerate(game_infos):
        m = mask_np[i]
        if m.sum() == 0:
            continue
        correct = int(np.array((preds[i] == tokens_j[i]) & jnp.array(m)).sum())
        a = correct / int(m.sum())
        per_game_acc.append((info["name"], float(a), int(m.sum())))
    for name, a, n in sorted(per_game_acc, key=lambda x: x[1])[:20]:
        print(f"  {name:35s}  acc={a:.3f}  ({n} tokens)")
    if len(per_game_acc) > 20:
        mean_acc = np.mean([a for _, a, _ in per_game_acc])
        print(f"  ... ({len(per_game_acc)} games total, mean acc={mean_acc:.3f})")

    out = {
        "enc_params": enc_p,
        "latent_params": latent_p,
        "dec_params": dec_p,
        "game_names": [info["name"] for info in game_infos],
        "raw_slots_all": np.array(raw_slots_all),
        "slots_all": np.array(slots_all),
        "vq_indices": None if vq_indices_final is None else vq_indices_final_np,
        "vq_stage_histograms": vq_stage_hist_final,
        "vq_joint_histogram": vq_joint_hist_final,
        "tokens": tokens_np,
        "mask": mask_np,
        "final_acc": float(acc_final),
        "final_loss": float(loss_final),
        "latent_aux_final": tuple(_tree_to_python(x) for x in latent_aux_final),
        "history": history,
        "per_game_acc": per_game_acc,
        "args": vars(args),
    }
    out_path = os.path.join(args.save_dir, "slot_ae.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    np.savez(
        os.path.join(args.save_dir, "history.npz"),
        **{k: np.asarray(v) for k, v in history.items()},
    )
    summary = {
        "latent_model": args.latent_model,
        "final_loss": float(loss_final),
        "final_acc": float(acc_final),
        "mean_per_game_acc": float(np.mean([a for _, a, _ in per_game_acc])),
        "min_per_game_acc": float(min(a for _, a, _ in per_game_acc)),
        "worst_game": min(per_game_acc, key=lambda x: x[1])[0],
        "n_games": len(per_game_acc),
        "n_updates": args.n_updates,
        "freeze_encoder": args.freeze_encoder,
        "init_from": args.init_from,
        "vocab_size": args.vocab_size,
        "latent_aux_final": tuple(_tree_to_python(x) for x in latent_aux_final),
        "args": vars(args),
    }
    if args.latent_model == "vae":
        summary.update({
            "kl_total": float(latent_aux_final[0]),
            "kl_per_dim": float(latent_aux_final[1]),
            "sigma_mean": float(latent_aux_final[2]),
            "loss_with_kl": float(loss_final + args.vae_kl_weight * latent_aux_final[0]),
        })
    elif args.latent_model == "vqvae":
        summary.update({
            "vq_codebook_loss": float(latent_aux_final[0]),
            "vq_commit_loss": float(latent_aux_final[1]),
            "vq_usage_loss": float(latent_aux_final[2]),
            "vq_soft_perplexity": float(latent_aux_final[3]),
            "vq_hard_util": float(latent_aux_final[4]),
            "vq_margin_loss": float(latent_aux_final[5]),
            "vq_mean_margin": float(latent_aux_final[6]),
            "vq_hard_balance_loss": float(latent_aux_final[7]),
            "vq_hard_balance_perp": float(latent_aux_final[8]),
            "vq_assign_temp": float(latent_aux_final[9]),
            "vq_assign_perplexity": float(latent_aux_final[10]),
            "vq_stage_hard_utils": _tree_to_python(latent_aux_final[11]),
            "vq_stage_mean_margins": _tree_to_python(latent_aux_final[12]),
            "vq_residual_norms": _tree_to_python(latent_aux_final[13]),
            "vq_joint_hard_util": float(latent_aux_final[4]),
            "vq_stage_histograms": vq_stage_hist_final,
            "vq_joint_histogram": vq_joint_hist_final,
            "vq_hard_histogram": vq_hist_final,
            "loss_with_vq": float(
                loss_final
                + args.vq_loss_weight * (
                    latent_aux_final[0]
                    + args.vq_commitment_weight * latent_aux_final[1]
                )
                + args.vq_usage_loss_weight * latent_aux_final[2]
                + args.vq_margin_loss_weight * latent_aux_final[5]
                + args.vq_hard_balance_loss_weight * latent_aux_final[7]
            ),
        })
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
