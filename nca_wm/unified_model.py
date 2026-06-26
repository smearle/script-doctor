"""Unified composable NCA world model: conditioning x recurrence on one body.

This is the single source-of-truth architecture that subsumes the four
historical Flax classes (``NCAWorldModel``, ``ConditionalNCAWorldModel``,
``RuleAttnNCAWorldModel``, ``RecurrentNCAWorldModel``) as points on a 2-axis
grid:

    cond      in {"none", "film", "rule_attn"}    # how the game spec conditions
    recurrent in {False, True}                     # carry hidden grid across ticks

The body is the shared "v2" structure that three of the four classes already
use *bit-identically*:

    Dense ``embed`` -> per-layer [ (pre-LN) -> 3x3 ``conv_{i}`` on [h, h_inp]
    -> optional pool features -> ``pool_proj_{i}`` -> (cond hook) -> GELU ->
    ``out_{i}`` -> residual -> mask ] -> ``readout`` / ``win_ln`` + ``win_out``.

Submodule names are chosen to match the originals exactly, so a checkpoint
trained by any of the existing classes loads into this module verbatim when
configured with the matching flags:

    cond="none",      recurrent=False  ==  NCAWorldModel
    cond="rule_attn", recurrent=False  ==  RuleAttnNCAWorldModel
    cond="none",      recurrent=True   ==  RecurrentNCAWorldModel
    cond="film",      recurrent=False  ==  (new) FiLM-on-v2 body
    cond=*,           recurrent=True   ==  (new) conditioned recurrence

The two *new* capabilities are FiLM/rule-attn conditioning composed with the
recurrent memory carry: the game latent (FiLM ``z`` or the K rule slots) is
computed once from ``(tokens, mask)`` and reused at every env tick.

NOTE: this is an additive module. The four legacy classes in ``models.py`` /
``rule_attn_model.py`` are left untouched (live training jobs import them);
encoders and helpers are reused here by import, never re-defined.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from nca_wm.models import N_ACTIONS, build_input_with_history, GameSpecEncoder
from nca_wm.rule_attn_model import RuleSlotEncoder, VectorQuantizer


def _pool_feats_inline(h, *, axis_pool, axis_cummax, global_pool):
    """Replicate the inline pool-feature list used by the v2 body verbatim.

    Order and op choice matter for param-shape compatibility: ``pool_proj_{i}``
    consumes ``concat([h_conv, *feats])`` so the feature count fixes its input
    width. This mirrors ``NCAWorldModel`` / ``RuleAttnNCAWorldModel`` /
    ``RecurrentNCAWorldModel`` exactly (NOT ``models._pool_features``, which the
    legacy FiLM body used with a different concat site).
    """
    feats = []
    if axis_pool:
        feats.append(jnp.broadcast_to(h.max(axis=2, keepdims=True), h.shape))
        feats.append(jnp.broadcast_to(h.max(axis=1, keepdims=True), h.shape))
    if axis_cummax:
        feats.append(jnp.maximum.accumulate(h, axis=2))
        feats.append(jnp.maximum.accumulate(h, axis=1))
    if global_pool:
        feats.append(jnp.broadcast_to(h.max(axis=(1, 2), keepdims=True), h.shape))
    return feats


class UnifiedNCAWorldModel(nn.Module):
    """Composable NCA world model: ``cond`` x ``recurrent`` on one v2 body."""

    # --- Axes ---
    cond: str = "none"          # "none" | "film" | "rule_attn"
    recurrent: bool = False

    # --- Body (shared across all variants) ---
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1              # set to max n_objs at init time
    n_repeats: int = 1
    use_layernorm: bool = False
    input_skip: bool = True
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    history: int = 0            # history channels (Option A); stateless only
    return_intermediates: bool = False

    # --- Recurrent axis ---
    bptt_window: int = 0        # 0 = full BPTT; >0 = truncate to last window ticks

    # --- rule_attn encoder / cross-attention ---
    vocab_size: int = 142
    max_seq_len: int = 192
    enc_d_model: int = 64
    enc_n_self_layers: int = 2
    n_slots: int = 16
    n_app_slots: int = 0
    d_slot: int = 64
    n_attn_heads: int = 4
    use_vq: bool = False
    vq_codebook_size: int = 512
    vq_commitment_weight: float = 0.25
    vq_entropy_temp: float = 1.0

    # --- film encoder ---
    d_model: int = 64
    n_heads: int = 4
    n_enc_layers: int = 2
    d_z: int = 64

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens=None, game_mask=None,
                 hist_states=None, hist_actions=None, z_override=None,
                 slots_override=None, cond_dropout_mask=None,
                 return_slots=False):
        """
        Stateless (recurrent=False):
            state:  (B, C, H, W)  action_onehot: (B, 5)
            -> logits (B, C, H, W), win_logit (B,), sprite_logits (B,C,5,5,4)
        Recurrent (recurrent=True):
            state:  (B, L, C, H, W)  action_onehot: (B, L, 5)
            -> logits (B, L, C, H, W), win_logits (B, L), sprite (B,L,C,5,5,4)

        Conditioning (``game_tokens`` / ``game_mask``) is required when
        ``cond != "none"`` and is encoded ONCE (slots for rule_attn, z for
        film), then reused across every NCA step / env tick.
        """
        # ------------------------------------------------------------------
        # 1. Encode the game spec -> conditioning signal (computed once).
        # ------------------------------------------------------------------
        slots_dyn = None        # rule_attn: (B, n_dyn, d_slot)
        gamma = beta = None     # film: (B, 1, 1, n_hid)
        all_slots = None

        if self.cond == "rule_attn":
            encoder = RuleSlotEncoder(
                vocab_size=self.vocab_size, max_seq_len=self.max_seq_len,
                d_model=self.enc_d_model, n_self_layers=self.enc_n_self_layers,
                n_slots=self.n_slots, d_slot=self.d_slot,
                n_heads=self.n_attn_heads, name="game_encoder",
            )
            encoded = encoder(game_tokens, game_mask)
            slots = slots_override if slots_override is not None else encoded
            if self.use_vq:
                vq = VectorQuantizer(
                    codebook_size=self.vq_codebook_size, d_slot=self.d_slot,
                    commitment_weight=self.vq_commitment_weight,
                    entropy_temp=self.vq_entropy_temp, name="slot_vq",
                )
                if slots_override is None:
                    slots = vq(slots)[0]
                else:
                    _ = vq(encoded)  # register params; keep override continuous
            all_slots = slots
            n_dyn = self.n_slots - self.n_app_slots
            slots_dyn = slots[:, :n_dyn, :]
            if cond_dropout_mask is not None:
                slots_dyn = jnp.where(
                    cond_dropout_mask[:, None, None], 0.0, slots_dyn)

        elif self.cond == "film":
            z = GameSpecEncoder(
                vocab_size=self.vocab_size, d_model=self.d_model,
                n_heads=self.n_heads, n_layers=self.n_enc_layers,
                d_z=self.d_z, max_seq_len=self.max_seq_len, name="game_encoder",
            )(game_tokens, game_mask)
            if z_override is not None:
                z = z_override
            if cond_dropout_mask is not None:
                z = jnp.where(cond_dropout_mask[:, None], 0.0, z)
            gamma = nn.Dense(self.n_hid, name="film_gamma",
                             kernel_init=nn.initializers.zeros)(z) + 1.0
            beta = nn.Dense(self.n_hid, name="film_beta",
                            kernel_init=nn.initializers.zeros)(z)
            gamma = gamma[:, None, None, :]
            beta = beta[:, None, None, :]

        # ------------------------------------------------------------------
        # 2. Allocate body submodules once (names match the legacy classes).
        # ------------------------------------------------------------------
        if self.n_steps % self.n_repeats != 0:
            raise ValueError(
                f"n_steps ({self.n_steps}) must be divisible by n_repeats "
                f"({self.n_repeats})")
        n_layers = self.n_steps // self.n_repeats

        embed = nn.Dense(self.n_hid, name="embed")
        step_norm = nn.LayerNorm(name="step_ln") if self.use_layernorm else None
        convs = [nn.Conv(self.n_hid, (3, 3), padding="SAME", name=f"conv_{i}")
                 for i in range(n_layers)]
        has_pool = self.axis_pool or self.axis_cummax or self.global_pool
        pool_projs = ([nn.Dense(self.n_hid, name=f"pool_proj_{i}")
                       for i in range(n_layers)]
                      if has_pool else [None] * n_layers)
        outs = [nn.Dense(self.n_hid, name=f"out_{i}") for i in range(n_layers)]

        if self.cond == "rule_attn":
            attn_lns = [nn.LayerNorm(name=f"attn_ln_{i}") for i in range(n_layers)]
            slot_lns = [nn.LayerNorm(name=f"slot_ln_{i}") for i in range(n_layers)]
            xattns = [nn.MultiHeadDotProductAttention(
                num_heads=self.n_attn_heads, qkv_features=self.n_hid,
                name=f"cell_slot_xattn_{i}") for i in range(n_layers)]
        else:
            attn_lns = slot_lns = xattns = [None] * n_layers

        carry_norm = (nn.GroupNorm(num_groups=1, name="carry_gn")
                      if self.recurrent else None)
        readout_layer = nn.Dense(self.n_out, name="readout")
        win_ln = nn.LayerNorm(name="win_ln")
        win_out = nn.Dense(1, name="win_out")

        # ------------------------------------------------------------------
        # 3. One NCA layer (shared by stateless + recurrent paths).
        # ------------------------------------------------------------------
        def layer(h, h_inp, mask_bcast, i):
            B, H, W, _ = h.shape
            h_step = step_norm(h) if step_norm is not None else h
            conv_in = (jnp.concatenate([h_step, h_inp], axis=-1)
                       if self.input_skip else h_step)
            h_conv = convs[i](conv_in)
            feats = _pool_feats_inline(
                h, axis_pool=self.axis_pool, axis_cummax=self.axis_cummax,
                global_pool=self.global_pool)
            if feats:
                h_conv = pool_projs[i](jnp.concatenate([h_conv] + feats, axis=-1))

            core = h_conv
            if self.cond == "rule_attn":
                h_flat = h.reshape(B, H * W, self.n_hid)
                attn_out = xattns[i](attn_lns[i](h_flat),
                                     slot_lns[i](slots_dyn),
                                     deterministic=True)
                core = h_conv + attn_out.reshape(B, H, W, self.n_hid)

            delta = outs[i](nn.gelu(core))
            if self.cond == "film":
                delta = gamma * delta + beta
            h = h + delta
            return h * mask_bcast

        def body(h, h_inp, mask_bcast):
            for _ in range(self.n_repeats):
                for i in range(n_layers):
                    h = layer(h, h_inp, mask_bcast, i)
            return h

        def win_head(h, mask_bcast):
            pooled = (h * mask_bcast).sum(axis=(1, 2)) / jnp.maximum(
                mask_bcast.sum(axis=(1, 2)), 1.0)
            return win_out(win_ln(pooled)).squeeze(-1)

        # ------------------------------------------------------------------
        # 4a. Recurrent path: carry h across ticks.
        # ------------------------------------------------------------------
        if self.recurrent:
            B, L, C, H, W = state.shape
            h = jnp.zeros((B, H, W, self.n_hid), dtype=jnp.float32)
            cut = (L - self.bptt_window) if self.bptt_window else 0
            logits_seq, win_seq = [], []
            for t in range(L):
                if self.bptt_window and t == cut:
                    h = jax.lax.stop_gradient(h)
                x = state[:, t].transpose(0, 2, 3, 1)
                mask_bcast = (x.sum(axis=-1, keepdims=True) > 0).astype(jnp.float32)
                act = jnp.broadcast_to(
                    action_onehot[:, t][:, None, None, :], (B, H, W, N_ACTIONS))
                h_inp = embed(jnp.concatenate([x, act], axis=-1))
                h = (h + h_inp) * mask_bcast
                h = body(h, h_inp, mask_bcast)
                h = carry_norm(h) * mask_bcast
                logits_seq.append(readout_layer(h).transpose(0, 3, 1, 2))
                win_seq.append(win_head(h, mask_bcast))
            logits = jnp.stack(logits_seq, axis=1)
            win_logits = jnp.stack(win_seq, axis=1)
            sprite = jnp.zeros((B, L, self.n_out, 5, 5, 4), dtype=jnp.float32)
            if return_slots:
                return logits, win_logits, sprite, all_slots
            return logits, win_logits, sprite

        # ------------------------------------------------------------------
        # 4b. Stateless path: single f(state, action) -> next.
        # ------------------------------------------------------------------
        B, C, H, W = state.shape
        x = state.transpose(0, 2, 3, 1)
        act = jnp.broadcast_to(action_onehot[:, None, None, :],
                               (B, H, W, N_ACTIONS))
        inp = build_input_with_history(x, act, hist_states, hist_actions)
        mask_bcast = (x.sum(axis=-1, keepdims=True) > 0).astype(jnp.float32)
        h_inp = embed(inp)
        h = h_inp * mask_bcast
        h = body(h, h_inp, mask_bcast)
        logits = readout_layer(h).transpose(0, 3, 1, 2)
        win_logit = win_head(h, mask_bcast)
        sprite = jnp.zeros((B, self.n_out, 5, 5, 4), dtype=jnp.float32)
        if return_slots:
            return logits, win_logit, sprite, all_slots
        return logits, win_logit, sprite
