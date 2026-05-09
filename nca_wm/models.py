"""NCA world model classes (unconditional + conditional rule-attn-style).

Lifted from train.py to keep that file focused on training-loop logic.
The historical NCAWorldModel (unconditional) and ConditionalNCAWorldModel
(FiLM-conditional with GameSpecEncoder) live here; the rule-attention NCA
itself remains in nca_wm/rule_attn_model.py and the non-NCA baselines
(CNN/UNet/ViT) in nca_wm/baselines.py.

Apply signature for cond / uncond models:

    NCAWorldModel.apply(params, state, action_onehot)
        -> (logits, win_logit, sprite_logits)

    ConditionalNCAWorldModel.apply(
        params, state, action_onehot, game_tokens, game_masks)
        -> (logits, win_logit, sprite_logits)

Padding cells (introduced by per-bucket batching of mixed-size levels)
are always masked from the hidden state and the win-pool — the mask is
derived from the input itself (real cells have >=1 channel set; bucket
padding is all-zero).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

# Action-space size mirrors train.py (kept here as a constant so this
# module doesn't depend on train.py at import time).
N_ACTIONS = 5

# ---------------------------------------------------------------------------
# 2. NCA world model (Flax/JAX)
# ---------------------------------------------------------------------------

def _pool_features(h, *, axis_pool: bool, axis_cummax: bool, global_pool: bool):
    """Augment NHWC hidden state with global-context features for use as
    extra input channels to the next NCA conv.

    All operations preserve the channel dim (one summary value per channel)
    and broadcast back to (B, H, W, C). No new parameters.

    - axis_pool   (2 features): max over W, max over H. Each cell sees
                  "max of channel c anywhere in my row / my column".
                  Needed for rules like `[ X | ... | Y ]` (X exists in row).
    - axis_cummax (4 features): prefix max from L→R, R→L, T→B, B→T along
                  each axis. Each cell sees "max of channel c to my left /
                  right / above / below". Encodes directional info that plain
                  axis_pool loses.
    - global_pool (1 feature): max over (H, W). Each cell sees "max of
                  channel c anywhere in the grid". Needed for multi-bracket
                  rules like `[X] [Y]` (X and Y both exist somewhere).
    """
    feats = []
    if axis_pool:
        row_max = jnp.max(h, axis=2, keepdims=True)  # (B, H, 1, C)
        col_max = jnp.max(h, axis=1, keepdims=True)  # (B, 1, W, C)
        feats.append(jnp.broadcast_to(row_max, h.shape))
        feats.append(jnp.broadcast_to(col_max, h.shape))
    if axis_cummax:
        feats.append(jax.lax.cummax(h, axis=2))                # L→R along W
        feats.append(jax.lax.cummax(h, axis=2, reverse=True))  # R→L
        feats.append(jax.lax.cummax(h, axis=1))                # T→B along H
        feats.append(jax.lax.cummax(h, axis=1, reverse=True))  # B→T
    if global_pool:
        gmax = jnp.max(h, axis=(1, 2), keepdims=True)          # (B, 1, 1, C)
        feats.append(jnp.broadcast_to(gmax, h.shape))
    if not feats:
        return None
    return jnp.concatenate(feats, axis=-1)


class NCAWorldModel(nn.Module):
    """Neural Cellular Automaton world model (unconditional).

    Mirrors :class:`RuleAttnNCAWorldModel`'s body structure exactly,
    minus the slot encoder + cross-attention path. This makes the
    cond-vs-uncond comparison apples-to-apples on body architecture:
    both run the same conv → pool-projection → residual-update structure,
    use the same Dense embed, GELU activation, optional LayerNorm
    pre-step, and per-step (or weight-shared) layer pattern. The only
    architectural differences are the encoder and cross-attention.

    Pool features (``axis_pool`` / ``axis_cummax`` / ``global_pool``) are
    computed from ``h`` and concatenated to the *output* of the 3×3 conv,
    then projected back to ``n_hid`` via a Dense layer — this is rule_attn's
    pattern, not the older "concat into conv input" approach. The latter
    inflates conv input width by ~6× and made gallery-scale grids OOM
    on 24 GiB cards at param counts that the cond body fits at.

    Padding cells (introduced by per-bucket batching of mixed-size levels)
    are always masked from the hidden state and the win-pool — the mask
    is derived from the input itself (real cells have ≥1 channel set;
    bucket padding is all-zero).
    """
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1  # set to n_objs at init time
    return_intermediates: bool = False
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    # LayerNorm on h pre-step (mirrors rule_attn). Off by default — enable
    # when running deeper (large n_nca_steps) models to stabilize training.
    use_layernorm: bool = False
    # Re-inject the embedded (state, action) at every NCA step. Mirrors
    # rule_attn's input_skip; consistently helpful at depth ≥ 8 per
    # ARCHITECTURE_REPORT F4.
    input_skip: bool = True
    # Factor `n_steps` into (n_layers × n_repeats) — n_layers distinct
    # weight sets, each applied n_repeats times. Default n_repeats=1 mirrors
    # the cond model's per-step body (no weight sharing across steps).
    n_repeats: int = 1

    @nn.compact
    def __call__(self, state, action_onehot):
        """
        Args:
            state: (B, C, H, W) float32 multihot level.
            action_onehot: (B, 5) float32 one-hot action.
        Returns:
            (logits, win_logit, sprite_logits) where
              logits: (B, C, H, W) next-state logits
              win_logit: (B,) scalar logit for P(next_state is winning)
              sprite_logits: zeros placeholder for tuple-shape parity
                with ConditionalNCAWorldModel / RuleAttnNCAWorldModel.
        """
        B, C, H, W = state.shape
        if self.n_steps % self.n_repeats != 0:
            raise ValueError(
                f"n_steps ({self.n_steps}) must be divisible by n_repeats "
                f"({self.n_repeats})"
            )
        n_layers = self.n_steps // self.n_repeats

        # NHWC for Flax convolutions
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)
        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act], axis=-1)  # (B, H, W, C+5)

        # 1. Dense embed (mirrors rule_attn)
        h_inp = nn.Dense(self.n_hid, name="embed")(inp)
        h = h_inp

        # 2. Mask: real cells have ≥1 channel set; padding is all-zero.
        mask_bcast = (x.sum(axis=-1, keepdims=True) > 0).astype(jnp.float32)
        h = h * mask_bcast

        # 3. Per-layer modules (n_layers distinct sets, applied n_repeats
        # times). Mirrors rule_attn's pre-instantiation pattern.
        step_norm = nn.LayerNorm(name="step_ln") if self.use_layernorm else None
        convs = [
            nn.Conv(self.n_hid, kernel_size=(3, 3), padding="SAME",
                    name=f"conv_{i}")
            for i in range(n_layers)
        ]
        has_pool = self.axis_pool or self.axis_cummax or self.global_pool
        pool_projs = (
            [nn.Dense(self.n_hid, name=f"pool_proj_{i}")
             for i in range(n_layers)]
            if has_pool else [None] * n_layers
        )
        outs = [nn.Dense(self.n_hid, name=f"out_{i}")
                for i in range(n_layers)]

        readout_layer = nn.Dense(self.n_out, name="readout")
        win_ln = nn.LayerNorm(name="win_ln")
        win_out = nn.Dense(1, name="win_out")

        hidden_steps = []
        readout_steps = []

        for r in range(self.n_repeats):
            for i in range(n_layers):
                h_step = step_norm(h) if step_norm is not None else h
                # (a) Conv on h alone (or [h, h_inp] with input_skip).
                conv_in = (jnp.concatenate([h_step, h_inp], axis=-1)
                           if self.input_skip else h_step)
                h_conv = convs[i](conv_in)

                # (b) Pool features from h, concat to conv output, Dense
                # project back to n_hid (mirrors rule_attn).
                pool_feats = []
                if self.axis_pool:
                    row_max = h.max(axis=2, keepdims=True)
                    col_max = h.max(axis=1, keepdims=True)
                    pool_feats.append(jnp.broadcast_to(row_max, h.shape))
                    pool_feats.append(jnp.broadcast_to(col_max, h.shape))
                if self.axis_cummax:
                    row_cummax = jnp.maximum.accumulate(h, axis=2)
                    col_cummax = jnp.maximum.accumulate(h, axis=1)
                    pool_feats.append(row_cummax)
                    pool_feats.append(col_cummax)
                if self.global_pool:
                    global_max = h.max(axis=(1, 2), keepdims=True)
                    pool_feats.append(jnp.broadcast_to(global_max, h.shape))
                if pool_feats:
                    pool_cat = jnp.concatenate([h_conv] + pool_feats, axis=-1)
                    h_conv = pool_projs[i](pool_cat)

                # (c) Residual update. No cross-attention path (vs rule_attn).
                delta = nn.gelu(h_conv)
                delta = outs[i](delta)
                h = h + delta
                h = h * mask_bcast

                if self.return_intermediates:
                    hidden_steps.append(h)
                    step_logits = readout_layer(h).transpose(0, 3, 1, 2)
                    readout_steps.append(step_logits)

        # 4. Readout: per-cell next-state logits.
        logits = readout_layer(h).transpose(0, 3, 1, 2)

        # 5. Win head: mask-weighted pool over real cells, LN, Dense.
        pooled = (h * mask_bcast).sum(axis=(1, 2)) / jnp.maximum(
            mask_bcast.sum(axis=(1, 2)), 1.0
        )
        pooled = win_ln(pooled)
        win_logit = win_out(pooled).squeeze(-1)

        # Sprite placeholder for tuple-shape parity.
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4), dtype=jnp.float32)

        if self.return_intermediates:
            return logits, win_logit, sprite_logits, {
                "hidden": hidden_steps, "readouts": readout_steps,
            }
        return logits, win_logit, sprite_logits


# ---------------------------------------------------------------------------
# 2a. Conditional NCA world model (game-spec encoder + FiLM)
# ---------------------------------------------------------------------------

class GameSpecEncoder(nn.Module):
    """Transformer encoder: game token sequence -> latent z.

    Prepends a learnable [CLS] token. Output is the CLS representation
    projected to d_z dimensions.
    """
    vocab_size: int = 142       # VOCAB_SIZE + 1 for CLS
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_z: int = 64
    max_seq_len: int = 192      # max tokens + 1 for CLS
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, token_ids, mask, deterministic=True):
        """
        Args:
            token_ids: (B, S) int32 token IDs (without CLS, PAD=0).
            mask: (B, S) bool, True for real tokens.
        Returns:
            z: (B, d_z) float32 latent vector.
        """
        B, S = token_ids.shape
        # Token + positional embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")

        # CLS token (position 0)
        cls_tok = jnp.full((B, 1), self.vocab_size - 1, dtype=jnp.int32)  # CLS token ID
        cls_mask = jnp.ones((B, 1), dtype=jnp.bool_)

        # Prepend CLS
        all_tokens = jnp.concatenate([cls_tok, token_ids], axis=1)  # (B, 1+S)
        all_mask = jnp.concatenate([cls_mask, mask], axis=1)        # (B, 1+S)

        L = all_tokens.shape[1]
        positions = jnp.arange(L)[None, :]  # (1, L)
        x = tok_emb(all_tokens) + pos_emb(positions)  # (B, L, d_model)

        # Attention mask: Flax's MHA treats mask as BOOLEAN (True=keep,
        # False=mask-out) via `jnp.where(mask, attn_weights, big_neg)`.
        # Prior version passed a float mask (0.0=keep, -1e9=mask-out), which
        # jnp.where interprets via truthy/falsy semantics — 0.0 is falsy, so
        # real tokens got masked OUT and padding got KEPT (inverted). This
        # collapsed all games to near-identical z's. Bool mask fixes it.
        attn_mask = all_mask[:, None, None, :]  # (B, 1, 1, L) bool

        # Transformer encoder layers
        for i in range(self.n_layers):
            # Pre-norm self-attention
            y = nn.LayerNorm(name=f"ln1_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                name=f"attn_{i}",
            )(y, y, mask=attn_mask, deterministic=deterministic)
            x = x + y
            # Pre-norm FFN
            y = nn.LayerNorm(name=f"ln2_{i}")(x)
            y = nn.Dense(self.d_model * 4, name=f"ff1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"ff2_{i}")(y)
            x = x + y

        x = nn.LayerNorm(name="ln_final")(x)

        # CLS output -> z
        cls_out = x[:, 0, :]  # (B, d_model)
        z = nn.Dense(self.d_z, name="z_proj")(cls_out)
        return z


class ConditionalNCAWorldModel(nn.Module):
    """NCA world model conditioned on a game specification via FiLM.

    The game spec (token sequence) is encoded by a transformer into a latent z.
    At each NCA step, z modulates the hidden state update via
    FiLM: dh = gamma(z) * dh + beta(z).

    Optional global-context flags (axis_pool, axis_cummax, global_pool) are
    identical to NCAWorldModel and inject pooled hidden features as extra
    conv inputs at each step.
    """
    # NCA params
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1
    return_intermediates: bool = False
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    use_layernorm: bool = False
    # Sprite decoder: small Dense head on z → (n_out, 5, 5, 4) RGBA kernel
    # per object (the "lookup table" per-game sprite set). When False,
    # sprite_logits output is zeros.
    sprite_decoder: bool = False
    sprite_hid: int = 128
    # Encoder params
    vocab_size: int = 142
    d_model: int = 64
    n_heads: int = 4
    n_enc_layers: int = 2
    d_z: int = 64
    max_seq_len: int = 192

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask,
                 z_override=None):
        """
        Args:
            state: (B, C, H, W) float32 multihot level.
            action_onehot: (B, 5) float32 one-hot action.
            game_tokens: (B, S) int32 tokenized game spec.
            game_mask: (B, S) bool mask (True for real tokens).
            z_override: optional (B, d_z) latent to substitute for the
                encoder's output. When provided, the FiLM/NCA path uses
                this z; the encoder is still invoked (with the supplied
                tokens) so its params are exercised, then its output is
                discarded. Used by interpolation/sampling tools to roll
                out under custom latents without rebuilding the module.
        Returns:
            (logits, win_logit) or (logits, win_logit, intermediates).
        """
        B, C, H, W = state.shape

        # --- Encode game spec → z ---
        z = GameSpecEncoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_enc_layers,
            d_z=self.d_z,
            max_seq_len=self.max_seq_len,
            name="game_encoder",
        )(game_tokens, game_mask)  # (B, d_z)
        if z_override is not None:
            z = z_override

        # --- FiLM parameters from z (shared across NCA steps) ---
        # Initialize gamma near 1, beta near 0 for identity-like start
        gamma = nn.Dense(
            self.n_hid, name="film_gamma",
            kernel_init=nn.initializers.zeros,
        )(z) + 1.0  # (B, n_hid)
        beta = nn.Dense(
            self.n_hid, name="film_beta",
            kernel_init=nn.initializers.zeros,
        )(z)  # (B, n_hid)

        # Broadcast for spatial dims: (B, 1, 1, n_hid)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        # --- NCA forward (same as NCAWorldModel, with FiLM) ---
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)

        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act], axis=-1)
        mask_bcast = (x.sum(axis=-1, keepdims=True) > 0).astype(jnp.float32)

        h = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="embed")(inp)
        h = nn.relu(h)
        h = h * mask_bcast

        nca_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME", name="nca_conv")
        nca_gate = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="nca_gate")
        readout_conv = nn.Conv(self.n_out, (1, 1), padding="SAME", name="readout")
        nca_norm = nn.LayerNorm(name="nca_norm") if self.use_layernorm else None

        hidden_steps = []
        readout_steps = []

        for _ in range(self.n_steps):
            parts = [h, inp]
            pool_feats = _pool_features(
                h,
                axis_pool=self.axis_pool,
                axis_cummax=self.axis_cummax,
                global_pool=self.global_pool,
            )
            if pool_feats is not None:
                parts.append(pool_feats)
            h_in = jnp.concatenate(parts, axis=-1)
            dh = nca_conv(h_in)
            dh = nn.relu(dh)
            dh = nca_gate(dh)
            # FiLM modulation on the update
            dh = gamma * dh + beta
            h = h + dh
            h = nn.relu(h)
            if nca_norm is not None:
                h = nca_norm(h)
            h = h * mask_bcast

            if self.return_intermediates:
                hidden_steps.append(h)
                step_logits = readout_conv(h).transpose(0, 3, 1, 2)
                readout_steps.append(step_logits)

        logits = readout_conv(h)
        logits = logits.transpose(0, 3, 1, 2)

        # Win head: pool NCA features, concat z (game-spec latent), MLP -> 1 logit
        win_feat = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="win_conv")(h)
        win_feat = nn.relu(win_feat)
        win_feat = (win_feat * mask_bcast).sum(axis=(1, 2)) / jnp.maximum(
            mask_bcast.sum(axis=(1, 2)), 1.0
        )      # (B, n_hid)
        win_feat = jnp.concatenate([win_feat, z], axis=-1)  # (B, n_hid + d_z)
        win_feat = nn.Dense(self.n_hid, name="win_dense")(win_feat)
        win_feat = nn.relu(win_feat)
        win_logit = nn.Dense(1, name="win_out")(win_feat)[:, 0]  # (B,)

        # Sprite decoder head: z → per-object 5x5x4 RGBA kernel.
        # Output shape (B, n_out, 5, 5, 4). Sigmoid applied downstream where
        # a normalized [0,1] value is required (e.g. for MSE vs target).
        # This is a pure per-game lookup table (no cross-channel weights),
        # implemented as a Dense head whose output is reshaped — equivalent
        # to a learned (n_out, 5, 5, 4) tensor produced from z.
        if self.sprite_decoder:
            sh = nn.Dense(self.sprite_hid, name="sprite_hid")(z)
            sh = nn.relu(sh)
            sprite_flat = nn.Dense(
                self.n_out * 5 * 5 * 4, name="sprite_out"
            )(sh)  # (B, n_out*5*5*4)
            sprite_logits = sprite_flat.reshape(B, self.n_out, 5, 5, 4)
        else:
            sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4), dtype=jnp.float32)

        if self.return_intermediates:
            return logits, win_logit, sprite_logits, {"hidden": hidden_steps, "readouts": readout_steps}
        return logits, win_logit, sprite_logits


