"""Autoregressive token decoder for FiLM-latent game autoencoding.

Closes the loop from (encoder → z) back to (z → token sequence), so we can
sample novel games by drawing z from an empirical or learned prior and
decoding to a token sequence (which can then be detokenized → PuzzleScript
source → played in the PS engine).

Non-destructive: this module shares the existing `GameSpecEncoder` from
train.py (reused by loading its params from a trained
ConditionalNCAWorldModel checkpoint). Nothing about NCA training changes.

Usage sketch:
    encoder_params = load_encoder_from_ckpt(nca_ckpt_path)
    decoder = TokenDecoder(vocab_size=142, max_seq_len=192, d_model=128,
                            n_layers=4, n_heads=4)
    params = decoder.init(rng, tokens, z, deterministic=True)
    # train with cross-entropy vs shifted tokens
    logits = decoder.apply(params, tokens_in, z)  # (B, L, vocab)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


class TokenDecoder(nn.Module):
    """Causal transformer decoder: latent z (+ teacher-forcing tokens) -> logits.

    The latent z is projected to a single 'memory' token and prepended to the
    token sequence. Standard causal self-attention then lets each position
    attend to (z, prior tokens). The final head predicts the next token at
    each position.
    """
    vocab_size: int = 142
    max_seq_len: int = 192
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, tokens_in, z, deterministic: bool = True):
        """
        Args:
            tokens_in: (B, L) int32 input tokens (usually targets shifted right,
                starting with a BOS/CLS-like token if you want one).
            z: (B, d_z) float32 latent vector from the encoder.
        Returns:
            logits: (B, L, vocab_size) float32 — next-token logits at each
                input position (position i predicts token i+1, standard LM).
        """
        B, L = tokens_in.shape
        # Project z to d_model, prepend as a single memory token
        z_tok = nn.Dense(self.d_model, name="z_proj")(z)[:, None, :]  # (B, 1, d_model)

        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")(tokens_in)
        pos = jnp.arange(L)[None, :]
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")(pos)
        tok_emb = tok_emb + pos_emb  # (B, L, d_model)

        # Prepend z memory token (position 0 of the attended sequence)
        x = jnp.concatenate([z_tok, tok_emb], axis=1)  # (B, L+1, d_model)
        Lp = L + 1

        # Causal mask: each query position q can attend to keys k where k <= q.
        # Flax attention treats mask as boolean (True=keep, False=mask out),
        # so pass the lower-triangular bool array directly. (Passing a float
        # mask gets re-interpreted via jnp.where's truthy/falsy semantics,
        # which makes 0.0=mask-out and -1e9=keep — the opposite of the
        # intended additive-bias convention.)
        causal = jnp.tril(jnp.ones((Lp, Lp), dtype=jnp.bool_))[None, None, :, :]

        for i in range(self.n_layers):
            y = nn.LayerNorm(name=f"ln1_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                name=f"attn_{i}",
            )(y, y, mask=causal, deterministic=deterministic)
            x = x + y
            y = nn.LayerNorm(name=f"ln2_{i}")(x)
            y = nn.Dense(self.d_model * 4, name=f"ff1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"ff2_{i}")(y)
            x = x + y

        x = nn.LayerNorm(name="ln_final")(x)
        # Drop the z-memory position — we only predict tokens, not z itself.
        x_tok = x[:, 1:, :]  # (B, L, d_model)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x_tok)
        return logits


class SlotTokenDecoder(nn.Module):
    """Causal transformer decoder conditioned on a K-slot game latent.

    Companion to RuleSlotEncoder (rule_attn_model.py). Each token position
    self-attends causally to prior positions AND cross-attends to the K
    slots at every layer. This matches the symmetry of RuleAttnNCAWorldModel
    (where each spatial cell cross-attends to slots at every NCA step), so
    the decoder uses the same conditioning shape the world model trained on.

    Use this instead of TokenDecoder when the upstream encoder produces a
    slot matrix (B, K, d_slot) rather than a single z (B, d_z).
    """
    vocab_size: int = 142
    max_seq_len: int = 192
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_slot: int = 64

    @nn.compact
    def __call__(self, tokens_in, slots, deterministic: bool = True):
        """
        Args:
            tokens_in: (B, L) int32 input tokens (right-shifted target).
            slots:    (B, K, d_slot) float32 slot matrix from RuleSlotEncoder.
        Returns:
            logits: (B, L, vocab_size).
        """
        B, L = tokens_in.shape
        # Token + positional embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")(tokens_in)
        pos = jnp.arange(L)[None, :]
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")(pos)
        x = tok_emb + pos_emb  # (B, L, d_model)

        # Project slots to d_model so cross-attn keys/values match.
        slots_proj = nn.Dense(self.d_model, name="slot_proj")(slots)

        # Lower-triangular causal self-attention mask (bool, True=keep).
        causal = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[None, None, :, :]

        for i in range(self.n_layers):
            # 1) Causal self-attention.
            y = nn.LayerNorm(name=f"ln1_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=self.d_model,
                name=f"self_attn_{i}",
            )(y, y, mask=causal, deterministic=deterministic)
            x = x + y
            # 2) Cross-attention to slots (no mask — all slots visible).
            y = nn.LayerNorm(name=f"ln2_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=self.d_model,
                name=f"cross_attn_{i}",
            )(y, slots_proj, deterministic=deterministic)
            x = x + y
            # 3) FFN.
            y = nn.LayerNorm(name=f"ln3_{i}")(x)
            y = nn.Dense(self.d_model * 4, name=f"ff1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"ff2_{i}")(y)
            x = x + y

        x = nn.LayerNorm(name="ln_final")(x)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x)
        return logits


def sample_tokens_from_slots(decoder, params, slots, max_len: int,
                              bos_id: int = 0, eos_id: int | None = None,
                              temperature: float = 0.0, rng=None):
    """Greedy/temperature sampling from a SlotTokenDecoder.

    Args:
        decoder: SlotTokenDecoder instance.
        params: trained decoder params.
        slots: (B, K, d_slot) float32.
        eos_id: if given, truncate each row at the first emitted EOS
            (inclusive) and zero out the rest. Use this when the decoder
            was trained with `--use_eos`. If None, no truncation is
            applied — the caller must clean up the trailing tokens.
    Returns:
        (B, max_len) int32 token sequence.
    """
    B = slots.shape[0]
    tokens = jnp.full((B, max_len), bos_id, dtype=jnp.int32)
    for t in range(max_len):
        logits = decoder.apply(params, tokens, slots, deterministic=True)
        step_logits = logits[:, t, :]
        if temperature <= 0:
            next_tok = jnp.argmax(step_logits, axis=-1)
        else:
            assert rng is not None, "temperature > 0 requires an rng"
            rng, sub = jax.random.split(rng)
            next_tok = jax.random.categorical(sub, step_logits / temperature, axis=-1)
        if t + 1 < max_len:
            tokens = tokens.at[:, t + 1].set(next_tok)
    if eos_id is not None:
        import numpy as _np
        toks_np = _np.asarray(tokens)
        cleaned = _np.zeros_like(toks_np)
        for b in range(B):
            row = toks_np[b]
            # Skip the leading BOS at position 0; find first EOS at position >= 1.
            after_bos = row[1:]
            hit = (after_bos == eos_id)
            first_eos = int(_np.argmax(hit)) if hit.any() else len(after_bos)
            # Keep positions 0 .. (1+first_eos) inclusive — i.e. through EOS.
            keep_through = 1 + first_eos + 1
            keep_through = min(keep_through, len(row))
            cleaned[b, :keep_through] = row[:keep_through]
        tokens = jnp.array(cleaned)
    return tokens


def decoder_loss(logits, targets, target_mask):
    """Cross-entropy loss on next-token prediction.

    Args:
        logits: (B, L, vocab) predictions at each position.
        targets: (B, L) int32 — target token at each position (i.e. the next
            token given inputs up to this position). Already shifted by the
            caller.
        target_mask: (B, L) bool — True for real target positions, False for
            padding. Loss averaged over True positions.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    nll = -(one_hot * log_probs).sum(-1)  # (B, L)
    mask = target_mask.astype(nll.dtype)
    loss = (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    # Per-token accuracy on unmasked positions
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets) & target_mask
    acc = correct.sum() / jnp.maximum(mask.sum(), 1.0)
    return loss, acc


def shift_right(tokens, bos_id: int = 0):
    """Shift a (B, L) int sequence right by one, filling position 0 with bos_id.
    Use as teacher-forcing input when tokens is the target sequence."""
    B, L = tokens.shape
    bos = jnp.full((B, 1), bos_id, dtype=tokens.dtype)
    return jnp.concatenate([bos, tokens[:, :-1]], axis=1)


def sample_tokens(decoder, params, encoder, encoder_params, z,
                   max_len: int, bos_id: int = 0, eos_id: int | None = None,
                   temperature: float = 1.0, rng=None):
    """Greedy/temperature sampling of a token sequence from a given z.

    Returns (B, max_len) int32 tokens. Stops each sample at the first EOS
    token if eos_id is given; otherwise generates exactly max_len tokens.
    """
    B = z.shape[0]
    tokens = jnp.full((B, max_len), bos_id, dtype=jnp.int32)
    for t in range(max_len):
        logits = decoder.apply(params, tokens, z, deterministic=True)
        step_logits = logits[:, t, :]  # (B, vocab)
        if temperature <= 0:
            next_tok = jnp.argmax(step_logits, axis=-1)
        else:
            assert rng is not None, "temperature > 0 requires an rng"
            rng, sub = jax.random.split(rng)
            next_tok = jax.random.categorical(sub, step_logits / temperature, axis=-1)
        if t + 1 < max_len:
            tokens = tokens.at[:, t + 1].set(next_tok)
    return tokens
