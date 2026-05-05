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


class VectorQuantizer(nn.Module):
    """VQ-VAE-style quantization of slot vectors against a learned codebook.

    Each input vector is replaced by its nearest codebook entry under squared
    L2 distance. Gradient through the quantization step uses the
    straight-through estimator (Oord et al., 2017), so the encoder can still
    be trained by the downstream task loss.

    Returns codebook and commitment losses separately; the caller is
    responsible for adding them (with a chosen commitment weight) to the
    total loss.
    """
    codebook_size: int = 512
    d_slot: int = 64
    commitment_weight: float = 0.25

    @nn.compact
    def __call__(self, slots):
        """
        slots: (..., d_slot) — typically (B, K, d_slot) for rule slots.
        Returns: (slots_q, codebook_loss, commitment_loss, indices)
            slots_q: same shape as slots, post-quantization (with STE).
            codebook_loss: scalar (pulls codebook entries to encoder outputs).
            commitment_loss: scalar (pulls encoder outputs to codebook entries).
            indices: same leading shape as slots, int32 codebook indices.
        """
        codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=1.0 / (self.d_slot ** 0.5)),
            (self.codebook_size, self.d_slot),
        )
        flat = slots.reshape(-1, self.d_slot)                  # (N, d)
        # squared L2 distance: ||x||^2 - 2 x·c + ||c||^2
        x_sq = jnp.sum(flat ** 2, axis=-1, keepdims=True)       # (N, 1)
        c_sq = jnp.sum(codebook ** 2, axis=-1)[None, :]         # (1, K_cb)
        xc   = flat @ codebook.T                                # (N, K_cb)
        dists = x_sq - 2.0 * xc + c_sq                          # (N, K_cb)
        idx_flat = jnp.argmin(dists, axis=-1)                    # (N,)
        quantized_flat = codebook[idx_flat]                      # (N, d)
        quantized = quantized_flat.reshape(slots.shape)
        indices = idx_flat.reshape(slots.shape[:-1])

        codebook_loss   = jnp.mean((jax.lax.stop_gradient(slots) - quantized) ** 2)
        commitment_loss = jnp.mean((slots - jax.lax.stop_gradient(quantized)) ** 2)

        # Straight-through: forward pass uses quantized; backward pass passes
        # gradient straight through to slots.
        slots_q = slots + jax.lax.stop_gradient(quantized - slots)
        return slots_q, codebook_loss, commitment_loss, indices


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
    n_slots: int = 16             # total slots K = dyn + app
    n_app_slots: int = 0          # last n_app_slots are app-only (decoder sees them, NCA does not)
    d_slot: int = 64
    # Cross-attention params (cells -> slots)
    n_attn_heads: int = 4
    # Global-context flags (same as ConditionalNCAWorldModel — reuse)
    axis_pool: bool = False
    axis_cummax: bool = False
    global_pool: bool = False
    # VQ-VAE-style codebook quantization on the encoder slots.
    # When use_vq=False (default), no VQ params or behaviour change.
    use_vq: bool = False
    vq_codebook_size: int = 512
    vq_commitment_weight: float = 0.25
    # Deep-unroll stabilization knobs. Both default-off so existing checkpoints
    # load unchanged. Enable when running large n_steps (≥8) where the bare
    # residual stack starts to diverge.
    #   use_layernorm: pre-norm LayerNorm on h at the start of every NCA step
    #     (one shared weight set across steps, like NCAWorldModel.use_layernorm).
    #   input_skip: re-inject the embedded input into the conv input at every
    #     step, mirroring NCAWorldModel's `parts = [h, inp]`. Makes it harder
    #     for deep stacks to lose track of the original observation.
    use_layernorm: bool = False
    input_skip: bool = False
    # Architectural padding mask. When True, the NCA's hidden state and
    # win-pool weighting are zeroed/excluded outside each sample's actual
    # playable area. The mask is derived from the input state itself: real
    # cells have at least the background bit set (sum across channels ≥ 1),
    # padding cells (added by per-bucket batching of mixed-size data) are
    # all-zero across channels (sum == 0). Helpful for multi-grid training
    # (closes a 5-7× transfer gap vs unmasked); harmful for single-grid
    # (causes the shared readout bias to collapse → degenerate "predict 0
    # everywhere"). Default off for safety; turn on explicitly when training
    # has multiple per-sample sizes (e.g. --synthetic_multi_grid).
    mask_hidden: bool = False
    # Factor `n_steps` into a (n_layers × n_repeats) hierarchy mirroring the
    # PuzzleScript engine's two-level loop:
    #   - Inner block of n_layers distinct rule-application layers (one full
    #     conv → pool → cross-attn → out per layer). Each layer has its own
    #     weights — analogous to the engine's ordered list of rules.
    #   - Outer iteration of n_repeats applications of that L-layer block,
    #     sharing weights across repeats. Analogous to the engine's `again`
    #     loop: same rule set, applied repeatedly until convergence.
    # Constraint: n_steps must be divisible by n_repeats; n_layers = n_steps //
    # n_repeats. Defaults (n_repeats=1) reproduce the historical per-step body
    # bit-identically. n_repeats=n_steps reproduces the old shared_weights
    # behaviour with names `conv_0` (vs the old `conv`); old shared
    # checkpoints will not load — there are none in active use.
    n_repeats: int = 1
    # Adaptive halting (PonderNet-style). When True, the model emits a
    # halting probability at every NCA step from a small head over the
    # globally-pooled `h`, plus a per-step readout (logits) and per-step
    # win logit. The training loop weights per-step losses by the cumulative
    # halt distribution and adds a KL-to-geometric-prior regularizer
    # (controlled by --halt_prior_p / --halt_kl_weight in train.py).
    # Default off; old checkpoints unaffected.
    adaptive_halt: bool = False

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask,
                 return_slots: bool = False, return_vq_aux: bool = False,
                 slots_override=None):
        """
        state: (B, C, H, W) multihot input.
        action_onehot: (B, N_ACTIONS).
        game_tokens: (B, S) int32.
        game_mask: (B, S) bool.
        return_slots: if True, also return the full (B, n_slots, d_slot) slot
            matrix (dyn + app) for downstream use (e.g. token decoder).
        slots_override: if provided ((B, n_slots, d_slot) float32), skip the
            encoder and use these slots directly. Used by inverse-fitting
            and latent-sampling tools to bypass the encoder. game_tokens /
            game_mask are still required for shape compatibility with the
            init/apply path but are unused on the forward pass.
        Returns: (logits, win_logit, sprite_logits[, all_slots])
          - logits: (B, C, H, W) next-state logits
          - win_logit: (B,)
          - sprite_logits: (B, C, 5, 5, 4) zeros placeholder
          - all_slots (only if return_slots=True): (B, n_slots, d_slot)
        """
        B, C, H, W = state.shape
        # 1. Encode game into K rule slots (or use override).
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
        if slots_override is not None:
            # Still call the encoder so its params are registered with this
            # module's variable scope (otherwise apply with the saved
            # checkpoint complains about unused params). The output is
            # discarded.
            _ = encoder(game_tokens, game_mask)
            slots = slots_override
        else:
            slots = encoder(game_tokens, game_mask)  # (B, K, d_slot), K = n_slots

        # Optional VQ-VAE quantization of slots against a shared codebook.
        # Gated entirely on use_vq so the default-off path is parameter- and
        # behaviour-identical to the pre-VQ model (existing checkpoints load
        # unchanged).
        if self.use_vq:
            vq = VectorQuantizer(
                codebook_size=self.vq_codebook_size,
                d_slot=self.d_slot,
                commitment_weight=self.vq_commitment_weight,
                name="slot_vq",
            )
            slots, vq_codebook_loss, vq_commitment_loss, vq_indices = vq(slots)
        else:
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            vq_codebook_loss = zero
            vq_commitment_loss = zero
            vq_indices = jnp.zeros(slots.shape[:-1], dtype=jnp.int32)

        # Split off appearance slots (last n_app_slots) — only the dyn slots
        # condition the NCA. The full slot tensor is returned for the
        # downstream token decoder.
        n_dyn = self.n_slots - self.n_app_slots
        slots_dyn = slots[:, :n_dyn, :]

        # 2. Embed input: (state, action) -> hidden state (B, H, W, n_hid)
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)
        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act], axis=-1)  # (B, H, W, C+5)
        h_inp = nn.Dense(self.n_hid, name="embed")(inp)
        h = h_inp

        # Spatial mask for padding cells (multi-grid / bucket-padded batches).
        # Derived from the input: real cells have ≥1 channel set; padding
        # cells are all-zero. mask: (B, H, W, 1) where 1 = real, 0 = padding.
        if self.mask_hidden:
            mask_2d = (x.sum(axis=-1) > 0).astype(jnp.float32)   # (B, H, W)
            mask_bcast = mask_2d[..., None]                       # (B, H, W, 1)
            h = h * mask_bcast
        else:
            mask_2d = None
            mask_bcast = None

        # Shared LayerNorm across NCA steps (pre-norm). Allocated once so
        # depth doesn't multiply parameter count of this stabilizer.
        step_norm = nn.LayerNorm(name="step_ln") if self.use_layernorm else None

        # Factor n_steps into n_layers (inner block) × n_repeats (outer loop,
        # weight-shared across repeats). Submodules for layer i are
        # instantiated once and re-applied n_repeats times — this is how
        # weight sharing works under Flax @nn.compact.
        if self.n_steps % self.n_repeats != 0:
            raise ValueError(
                f"n_steps ({self.n_steps}) must be divisible by n_repeats "
                f"({self.n_repeats}); got n_layers="
                f"{self.n_steps / self.n_repeats}"
            )
        n_layers = self.n_steps // self.n_repeats

        # Pre-instantiate the n_layers distinct submodules. Each list slot
        # is one full conv→pool→xattn→out layer; in n_repeats=1 these match
        # the historical per-step allocation bit-identically.
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
        attn_lns = [nn.LayerNorm(name=f"attn_ln_{i}")
                    for i in range(n_layers)]
        slot_lns = [nn.LayerNorm(name=f"slot_ln_{i}")
                    for i in range(n_layers)]
        xattns = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.n_attn_heads,
                qkv_features=self.n_hid,
                name=f"cell_slot_xattn_{i}",
            )
            for i in range(n_layers)
        ]
        outs = [nn.Dense(self.n_hid, name=f"out_{i}")
                for i in range(n_layers)]

        # Per-step readout / win / halt heads. The readout and win heads are
        # the same Modules used at the end of the model; lifting them above
        # the loop lets us call them at every step under adaptive_halt
        # without changing the un-shared / non-halt code path's parameters.
        # Halt head only allocated when adaptive_halt — keeps default-off
        # checkpoints param-identical.
        readout_layer = nn.Dense(self.n_out, name="readout")
        win_ln = nn.LayerNorm(name="win_ln")
        win_out = nn.Dense(1, name="win_out")
        if self.adaptive_halt:
            halt_ln = nn.LayerNorm(name="halt_ln")
            halt_out = nn.Dense(1, name="halt_out")

        # Buffers for per-step outputs (only used when adaptive_halt).
        per_step_logits = []   # each (B, n_out, H, W)
        per_step_win = []      # each (B,)
        per_step_halt = []     # each (B,) — raw logit, sigmoid → halt prob

        # 3. NCA steps. Each step:
        #    (a) 3x3 conv over hidden state (neighbor interaction)
        #    (b) optional global pool concatenation
        #    (c) cross-attention from each cell to K rule slots
        #    (d) residual update
        for r in range(self.n_repeats):
            for i in range(n_layers):
                # Pre-norm on h before the step (stabilizes deep unrolls).
                h_step = step_norm(h) if step_norm is not None else h
                # Optional input skip: re-inject the embedded (state, action)
                # so the model doesn't drift from the original observation
                # across many steps.
                conv_in = (jnp.concatenate([h_step, h_inp], axis=-1)
                           if self.input_skip else h_step)
                # (a) conv over hidden state
                h_conv = convs[i](conv_in)

                # (b) pool features (if requested). Same semantics as in
                # ConditionalNCAWorldModel; kept inline to avoid the
                # cross-module import cycle.
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
                    h_conv = pool_projs[i](pool_cat)

                # (c) cross-attention: each cell attends to rule slots.
                h_flat = h.reshape(B, H * W, self.n_hid)
                h_ln = attn_lns[i](h_flat)
                slots_dyn_ln = slot_lns[i](slots_dyn)
                attn_out = xattns[i](h_ln, slots_dyn_ln, deterministic=True)
                attn_out = attn_out.reshape(B, H, W, self.n_hid)

                # (d) residual update: conv path + slot attention path.
                delta = nn.gelu(h_conv + attn_out)
                delta = outs[i](delta)
                h = h + delta
                if mask_bcast is not None:
                    h = h * mask_bcast

                if self.adaptive_halt:
                    step_readout = readout_layer(h)         # (B, H, W, n_out)
                    step_logits = step_readout.transpose(0, 3, 1, 2)
                    step_pool = h.mean(axis=(1, 2))         # (B, n_hid)
                    step_win = win_out(win_ln(step_pool)).squeeze(-1)
                    step_halt = halt_out(halt_ln(step_pool)).squeeze(-1)
                    per_step_logits.append(step_logits)
                    per_step_win.append(step_win)
                    per_step_halt.append(step_halt)

        # 4. Readouts
        # Next-state logits
        readout = readout_layer(h)                          # (B, H, W, n_out)
        logits = readout.transpose(0, 3, 1, 2)               # (B, n_out, H, W)

        # Win logit: pool + dense. When mask_hidden is on, average over real
        # cells only (padded cells are 0 in h, but dividing by total H*W would
        # under-weight pooled signal in small samples).
        if mask_bcast is not None:
            pooled = (h * mask_bcast).sum(axis=(1, 2)) / jnp.maximum(
                mask_bcast.sum(axis=(1, 2)), 1.0
            )  # (B, n_hid)
        else:
            pooled = h.mean(axis=(1, 2))  # (B, n_hid)
        pooled = win_ln(pooled)
        win_logit = win_out(pooled).squeeze(-1)             # (B,)

        # Sprite placeholder (signature-compatible with ConditionalNCAWorldModel).
        sprite_logits = jnp.zeros((B, self.n_out, 5, 5, 4))

        vq_aux = (vq_codebook_loss, vq_commitment_loss, vq_indices)

        # When adaptive_halt is on, append per-step (logits, win, halt_logit)
        # as the final return element. When off, return shapes are
        # bit-identical to the pre-2026-05-03 model so existing callers
        # need no changes.
        if self.adaptive_halt:
            halt_logits_per_step = jnp.stack(per_step_halt, axis=0)  # (T, B)
            logits_per_step = jnp.stack(per_step_logits, axis=0)     # (T, B, n_out, H, W)
            win_per_step = jnp.stack(per_step_win, axis=0)           # (T, B)
            halt_aux = (logits_per_step, win_per_step, halt_logits_per_step)
            if return_slots and return_vq_aux:
                return logits, win_logit, sprite_logits, slots, vq_aux, halt_aux
            if return_slots:
                return logits, win_logit, sprite_logits, slots, halt_aux
            if return_vq_aux:
                return logits, win_logit, sprite_logits, vq_aux, halt_aux
            return logits, win_logit, sprite_logits, halt_aux

        # adaptive_halt = False — original return signatures.
        if return_slots and return_vq_aux:
            return logits, win_logit, sprite_logits, slots, vq_aux
        if return_slots:
            return logits, win_logit, sprite_logits, slots
        if return_vq_aux:
            return logits, win_logit, sprite_logits, vq_aux
        return logits, win_logit, sprite_logits
