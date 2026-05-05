"""Non-NCA world-model baselines for the rule_attn comparison.

Each baseline matches the apply signature of `RuleAttnNCAWorldModel`:

    model.apply(params, state, action_onehot, game_tokens, game_mask)
        -> (logits, win_logit, sprite_logits[, slots][, vq_aux])

So the same train.py / heldout_eval.py code paths drive them with no special
casing apart from model construction at `--architecture {cnn, unet, vit}`.

The baselines reuse `RuleSlotEncoder` from `rule_attn_model` verbatim, so the
comparison isolates the spatial-update body, not the conditioner. They share
a `_SharedHeads` readout (Dense for next-state logits, LN+Dense for win
logit) and the same `mask_hidden` semantics as the NCA: when on, padded
cells (all-zero across channels) are zeroed throughout the body and excluded
from the win-pool average.

`adaptive_halt` and `vq_codebook` are not supported by the baselines —
adaptive_halt has no analog in non-iterative models, and VQ on the slot
encoder is orthogonal but kept rule_attn-only for now to keep param counts
comparable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from nca_wm.rule_attn_model import RuleSlotEncoder


N_ACTIONS = 5


def _embed_input(state, action_onehot, n_hid, name="embed"):
    """(B, C, H, W) + (B, A) -> (B, H, W, n_hid). Mirrors RuleAttnNCAWorldModel."""
    B, C, H, W = state.shape
    x = state.transpose(0, 2, 3, 1)
    act = action_onehot[:, None, None, :]
    act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
    inp = jnp.concatenate([x, act], axis=-1)
    return nn.Dense(n_hid, name=name)(inp), x


def _padding_mask(state):
    """Real-cell mask from the multihot input: real if any channel set."""
    x = state.transpose(0, 2, 3, 1)
    m2 = (x.sum(axis=-1) > 0).astype(jnp.float32)
    return m2, m2[..., None]


def _heads(h, n_out, mask_bcast):
    """Shared readout + win pooling. Match RuleAttnNCAWorldModel exactly."""
    logits = nn.Dense(n_out, name="readout")(h).transpose(0, 3, 1, 2)
    if mask_bcast is not None:
        pooled = (h * mask_bcast).sum(axis=(1, 2)) / jnp.maximum(
            mask_bcast.sum(axis=(1, 2)), 1.0
        )
    else:
        pooled = h.mean(axis=(1, 2))
    pooled = nn.LayerNorm(name="win_ln")(pooled)
    win_logit = nn.Dense(1, name="win_out")(pooled).squeeze(-1)
    return logits, win_logit


def _wrap_returns(logits, win_logit, sprite_logits, slots,
                  return_slots, return_vq_aux):
    z = jnp.asarray(0.0, dtype=jnp.float32)
    vq_aux = (z, z, jnp.zeros(slots.shape[:-1], dtype=jnp.int32))
    if return_slots and return_vq_aux:
        return logits, win_logit, sprite_logits, slots, vq_aux
    if return_slots:
        return logits, win_logit, sprite_logits, slots
    if return_vq_aux:
        return logits, win_logit, sprite_logits, vq_aux
    return logits, win_logit, sprite_logits


def _slot_encoder_module(self):
    return RuleSlotEncoder(
        vocab_size=self.vocab_size,
        max_seq_len=self.max_seq_len,
        d_model=self.enc_d_model,
        n_self_layers=self.enc_n_self_layers,
        n_slots=self.n_slots,
        d_slot=self.d_slot,
        n_heads=self.n_attn_heads,
        name="game_encoder",
    )


# ----------------------------------------------------------------------------
# CNN baseline: deep ResNet, no iteration, no shared weights.
# Same per-block structure as one NCA layer (conv + optional pool + slot
# x-attn + residual gate), but `n_blocks` distinct layers applied once.
# Tests: how much does the iterated/shared structure of the NCA actually buy?
# ----------------------------------------------------------------------------
class CNNWorldModel(nn.Module):
    n_hid: int = 128
    n_out: int = 1
    n_blocks: int = 4

    # Encoder params (same as RuleAttnNCAWorldModel)
    vocab_size: int = 142
    max_seq_len: int = 192
    enc_d_model: int = 64
    enc_n_self_layers: int = 2
    n_slots: int = 16
    n_app_slots: int = 0
    d_slot: int = 64
    n_attn_heads: int = 4

    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    mask_hidden: bool = False

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask,
                 return_slots: bool = False, return_vq_aux: bool = False):
        B, C, H, W = state.shape
        slots = _slot_encoder_module(self)(game_tokens, game_mask)
        n_dyn = self.n_slots - self.n_app_slots
        slots_dyn = slots[:, :n_dyn, :]

        h, _x = _embed_input(state, action_onehot, self.n_hid, name="embed")
        if self.mask_hidden:
            _, mb = _padding_mask(state)
            h = h * mb
        else:
            mb = None

        has_pool = self.axis_pool or self.axis_cummax or self.global_pool
        for i in range(self.n_blocks):
            h_n = nn.LayerNorm(name=f"ln_{i}")(h)
            h_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME",
                             name=f"conv_{i}")(h_n)
            if has_pool:
                feats = [h_conv]
                if self.axis_pool:
                    feats.append(jnp.broadcast_to(h.max(axis=2, keepdims=True), h.shape))
                    feats.append(jnp.broadcast_to(h.max(axis=1, keepdims=True), h.shape))
                if self.axis_cummax:
                    feats.append(jnp.maximum.accumulate(h, axis=2))
                    feats.append(jnp.maximum.accumulate(h, axis=1))
                if self.global_pool:
                    feats.append(jnp.broadcast_to(h.max(axis=(1, 2), keepdims=True), h.shape))
                h_conv = nn.Dense(self.n_hid, name=f"pool_proj_{i}")(jnp.concatenate(feats, axis=-1))
            h_flat = h_n.reshape(B, H * W, self.n_hid)
            slots_n = nn.LayerNorm(name=f"slot_ln_{i}")(slots_dyn)
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.n_attn_heads, qkv_features=self.n_hid,
                name=f"xattn_{i}",
            )(h_flat, slots_n, deterministic=True)
            attn = attn.reshape(B, H, W, self.n_hid)
            delta = nn.Dense(self.n_hid, name=f"out_{i}")(nn.gelu(h_conv + attn))
            h = h + delta
            if mb is not None:
                h = h * mb

        logits, win_logit = _heads(h, self.n_out, mb)
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4))
        return _wrap_returns(logits, win_logit, sprite_logits, slots,
                             return_slots, return_vq_aux)


# ----------------------------------------------------------------------------
# U-Net baseline: 2-level encoder/decoder with skip connections; FiLM at the
# bottleneck from pooled rule slots. Standard grid-to-grid baseline.
# Tests: does multi-scale hierarchy + skip routing beat the NCA's flat
# repeated-local-update inductive bias?
# ----------------------------------------------------------------------------
class UNetWorldModel(nn.Module):
    n_hid: int = 128
    n_out: int = 1
    n_levels: int = 2
    n_bottleneck: int = 2

    vocab_size: int = 142
    max_seq_len: int = 192
    enc_d_model: int = 64
    enc_n_self_layers: int = 2
    n_slots: int = 16
    n_app_slots: int = 0
    d_slot: int = 64
    n_attn_heads: int = 4

    mask_hidden: bool = False

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask,
                 return_slots: bool = False, return_vq_aux: bool = False):
        B, C, H, W = state.shape
        slots = _slot_encoder_module(self)(game_tokens, game_mask)
        n_dyn = self.n_slots - self.n_app_slots
        slots_dyn = slots[:, :n_dyn, :]

        h, _x = _embed_input(state, action_onehot, self.n_hid, name="embed")
        if self.mask_hidden:
            _, mb = _padding_mask(state)
            h = h * mb
        else:
            mb = None

        skips = []
        cur = h
        for lvl in range(self.n_levels):
            cur = nn.gelu(nn.Conv(self.n_hid, (3, 3), padding="SAME",
                                   name=f"down_conv_{lvl}")(cur))
            skips.append(cur)
            cur = nn.gelu(nn.Conv(self.n_hid, (3, 3), strides=(2, 2),
                                   padding="SAME", name=f"down_pool_{lvl}")(cur))

        pooled = slots_dyn.mean(axis=1)
        gamma = nn.Dense(self.n_hid, name="film_gamma")(pooled)[:, None, None, :]
        beta = nn.Dense(self.n_hid, name="film_beta")(pooled)[:, None, None, :]
        for j in range(self.n_bottleneck):
            cn = nn.LayerNorm(name=f"bn_ln_{j}")(cur)
            cur = cur + nn.Conv(self.n_hid, (3, 3), padding="SAME",
                                 name=f"bn_conv_{j}")(nn.gelu(gamma * cn + beta))

        for lvl in reversed(range(self.n_levels)):
            cur = jax.image.resize(
                cur,
                (B, cur.shape[1] * 2, cur.shape[2] * 2, self.n_hid),
                method="nearest",
            )
            sk = skips[lvl]
            cur = cur[:, :sk.shape[1], :sk.shape[2], :]
            cur = jnp.concatenate([cur, sk], axis=-1)
            cur = nn.gelu(nn.Conv(self.n_hid, (3, 3), padding="SAME",
                                    name=f"up_conv_{lvl}")(cur))

        if mb is not None:
            cur = cur * mb

        logits, win_logit = _heads(cur, self.n_out, mb)
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4))
        return _wrap_returns(logits, win_logit, sprite_logits, slots,
                             return_slots, return_vq_aux)


# ----------------------------------------------------------------------------
# ViT baseline: flatten the grid to a sequence; apply N transformer encoder
# layers with self-attention over cells + cross-attention to rule slots +
# learned 2D positional embedding. Full global receptive field per layer.
# Tests: does the NCA's local-conv inductive bias beat unrestricted global
# attention at the same depth/width?
# ----------------------------------------------------------------------------
class ViTWorldModel(nn.Module):
    n_hid: int = 128
    n_out: int = 1
    n_layers: int = 4
    n_heads: int = 4
    max_grid: int = 64

    vocab_size: int = 142
    max_seq_len: int = 192
    enc_d_model: int = 64
    enc_n_self_layers: int = 2
    n_slots: int = 16
    n_app_slots: int = 0
    d_slot: int = 64
    n_attn_heads: int = 4

    mask_hidden: bool = False

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask,
                 return_slots: bool = False, return_vq_aux: bool = False):
        B, C, H, W = state.shape
        slots = _slot_encoder_module(self)(game_tokens, game_mask)
        n_dyn = self.n_slots - self.n_app_slots
        slots_dyn = slots[:, :n_dyn, :]

        h, _x = _embed_input(state, action_onehot, self.n_hid, name="embed")
        if self.mask_hidden:
            mask2, mb = _padding_mask(state)
            h = h * mb
        else:
            mask2, mb = None, None

        row_pos = self.param("row_pos", nn.initializers.normal(stddev=0.02),
                              (self.max_grid, self.n_hid))
        col_pos = self.param("col_pos", nn.initializers.normal(stddev=0.02),
                              (self.max_grid, self.n_hid))
        pos = row_pos[:H, None, :] + col_pos[None, :W, :]
        h = h + pos[None]

        seq = h.reshape(B, H * W, self.n_hid)
        if mask2 is not None:
            keep = mask2.reshape(B, H * W).astype(bool)
            sa_mask = keep[:, None, None, :]
        else:
            sa_mask = None

        for i in range(self.n_layers):
            sn = nn.LayerNorm(name=f"sa_ln_{i}")(seq)
            sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=self.n_hid,
                name=f"sa_{i}",
            )(sn, sn, mask=sa_mask, deterministic=True)
            seq = seq + sa
            cn = nn.LayerNorm(name=f"ca_ln_{i}")(seq)
            slots_n = nn.LayerNorm(name=f"slot_ln_{i}")(slots_dyn)
            ca = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=self.n_hid,
                name=f"ca_{i}",
            )(cn, slots_n, deterministic=True)
            seq = seq + ca
            fn = nn.LayerNorm(name=f"ff_ln_{i}")(seq)
            ff = nn.Dense(self.n_hid * 4, name=f"ff1_{i}")(fn)
            ff = nn.gelu(ff)
            ff = nn.Dense(self.n_hid, name=f"ff2_{i}")(ff)
            seq = seq + ff

        h = seq.reshape(B, H, W, self.n_hid)
        if mb is not None:
            h = h * mb

        logits, win_logit = _heads(h, self.n_out, mb)
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4))
        return _wrap_returns(logits, win_logit, sprite_logits, slots,
                             return_slots, return_vq_aux)
