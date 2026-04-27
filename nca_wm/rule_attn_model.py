"""Rule-attention architecture: perceiver-style slot encoder + NCA with
per-step cross-attention to rule slots.

Parallel to ConditionalNCAWorldModel (which uses a single pooled FiLM vector
as conditioning). Here the "game latent" is a K × d matrix of rule slots;
the NCA attends into those slots at every step so conditioning remains
expressive throughout the rollout.

Non-destructive: standalone module, reuses nothing from train.py except the
N_ACTIONS constant. Selected at the model-construction site via a CLI flag.

Architecture sketch:
  1. RuleSlotEncoder: game tokens (B, S) -> rule slots (B, K, d_slot).
     - Optionally a few self-attention layers on the tokens first.
     - Then K learned query vectors cross-attend to the token sequence.
     - Output is (B, K, d_slot).
  2. RuleAttnNCAWorldModel: at each NCA step, each cell's hidden state
     queries the K slots via cross-attention. The attention output is
     merged into the cell update alongside the usual conv-based neighbor
     interaction.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


N_ACTIONS = 5


class RuleSlotEncoder(nn.Module):
    """Perceiver-style encoder: tokens -> K × d_slot rule slots.

    K learned query vectors cross-attend to the tokenized game description
    to produce a fixed-size latent that scales sub-linearly with game size.
    """
    vocab_size: int = 142
    max_seq_len: int = 192
    d_model: int = 64            # token/feature embedding dim
    n_self_layers: int = 2        # depth of token self-attn before cross-attn
    n_slots: int = 16             # K
    d_slot: int = 64              # per-slot dim (= output channel dim)
    n_heads: int = 4

    @nn.compact
    def __call__(self, token_ids, mask, deterministic: bool = True):
        """
        token_ids: (B, S) int32
        mask: (B, S) bool — True for real tokens.
        Returns: (B, K, d_slot) float32 slot matrix.
        """
        B, S = token_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")(token_ids)
        pos = jnp.arange(S)[None, :]
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")(pos)
        x = tok_emb + pos_emb  # (B, S, d_model)

        # Self-attention over tokens (contextualize each rule token).
        # Bool mask semantics: True = keep, False = mask out.
        tok_mask = mask[:, None, None, :]  # (B, 1, 1, S)
        for i in range(self.n_self_layers):
            y = nn.LayerNorm(name=f"tok_ln1_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                name=f"tok_attn_{i}",
            )(y, y, mask=tok_mask, deterministic=deterministic)
            x = x + y
            y = nn.LayerNorm(name=f"tok_ln2_{i}")(x)
            y = nn.Dense(self.d_model * 4, name=f"tok_ff1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"tok_ff2_{i}")(y)
            x = x + y
        x = nn.LayerNorm(name="tok_ln_final")(x)

        # K learned query vectors (broadcast to batch).
        slot_queries = self.param(
            "slot_queries", nn.initializers.normal(stddev=0.02),
            (self.n_slots, self.d_slot),
        )
        q = jnp.broadcast_to(slot_queries[None], (B, self.n_slots, self.d_slot))

        # Cross-attention: queries = learned slots; keys/values = contextualized
        # token embeddings. Project token dim to slot dim via Dense inside MHA.
        slots = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_slot,
            name="slot_xattn",
        )(q, x, mask=tok_mask, deterministic=deterministic)
        slots = nn.LayerNorm(name="slot_ln")(slots)
        return slots  # (B, K, d_slot)


class RuleAttnNCAWorldModel(nn.Module):
    """NCA world model with per-step cross-attention to rule slots.

    At each NCA step, each spatial cell's hidden state attends to the K
    rule slots (computed once per game from RuleSlotEncoder). This lets
    different cells use different subsets of the game's rule structure at
    each timestep — richer than a single FiLM vector modulating all cells
    uniformly.
    """
    # NCA params
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1
    # Rule-slot encoder params
    vocab_size: int = 142
    max_seq_len: int = 192
    enc_d_model: int = 64
    enc_n_self_layers: int = 2
    n_slots: int = 16
    d_slot: int = 64
    # Cross-attention params (cells -> slots)
    n_attn_heads: int = 4
    # Global-context flags (same as ConditionalNCAWorldModel — reuse)
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask):
        """
        state: (B, C, H, W) multihot input.
        action_onehot: (B, N_ACTIONS).
        game_tokens: (B, S) int32.
        game_mask: (B, S) bool.
        Returns: (logits, win_logit, sprite_logits_placeholder)
          - logits: (B, C, H, W) next-state logits
          - win_logit: (B,)
          - sprite_logits: (B, C, 5, 5, 4) zeros placeholder
        """
        B, C, H, W = state.shape
        # 1. Encode game into K rule slots
        encoder = RuleSlotEncoder(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.enc_d_model,
            n_self_layers=self.enc_n_self_layers,
            n_slots=self.n_slots,
            d_slot=self.d_slot,
            n_heads=self.n_attn_heads,
            name="game_encoder",
        )
        slots = encoder(game_tokens, game_mask)  # (B, K, d_slot)

        # 2. Embed input: (state, action) -> hidden state (B, H, W, n_hid)
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)
        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act], axis=-1)  # (B, H, W, C+5)
        h = nn.Dense(self.n_hid, name="embed")(inp)

        # 3. NCA steps. Each step:
        #    (a) 3x3 conv over hidden state (neighbor interaction)
        #    (b) optional global pool concatenation
        #    (c) cross-attention from each cell to K rule slots
        #    (d) residual update
        for i in range(self.n_steps):
            # (a) conv over hidden state
            h_conv = nn.Conv(
                self.n_hid, kernel_size=(3, 3), padding="SAME", name=f"conv_{i}"
            )(h)

            # (b) pool features (if requested). Same semantics as in
            # ConditionalNCAWorldModel; kept inline to avoid the cross-module
            # import cycle.
            pool_feats = []
            if self.axis_pool:
                row_max = h.max(axis=2, keepdims=True)  # (B, H, 1, n_hid)
                col_max = h.max(axis=1, keepdims=True)  # (B, 1, W, n_hid)
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
                h_conv = nn.Dense(self.n_hid, name=f"pool_proj_{i}")(pool_cat)

            # (c) cross-attention: each cell attends to rule slots.
            # Flatten spatial: (B, H*W, n_hid).
            h_flat = h.reshape(B, H * W, self.n_hid)
            # LN inputs to attention (pre-norm style).
            h_ln = nn.LayerNorm(name=f"attn_ln_{i}")(h_flat)
            slots_ln = nn.LayerNorm(name=f"slot_ln_{i}")(slots)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_attn_heads,
                qkv_features=self.n_hid,
                name=f"cell_slot_xattn_{i}",
            )(h_ln, slots_ln, deterministic=True)  # (B, H*W, n_hid)
            attn_out = attn_out.reshape(B, H, W, self.n_hid)

            # (d) residual update: conv path + slot attention path.
            delta = nn.gelu(h_conv + attn_out)
            delta = nn.Dense(self.n_hid, name=f"out_{i}")(delta)
            h = h + delta

        # 4. Readouts
        # Next-state logits
        readout = nn.Dense(self.n_out, name="readout")(h)   # (B, H, W, n_out)
        logits = readout.transpose(0, 3, 1, 2)               # (B, n_out, H, W)

        # Win logit: pool + dense
        pooled = h.mean(axis=(1, 2))  # (B, n_hid)
        pooled = nn.LayerNorm(name="win_ln")(pooled)
        win_logit = nn.Dense(1, name="win_out")(pooled).squeeze(-1)  # (B,)

        # Sprite placeholder (signature-compatible with ConditionalNCAWorldModel).
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4))

        return logits, win_logit, sprite_logits
