"""Encoder-decoder event world model, geometry-generic.

Port of ``infogain-world-models`` ``mario_milestone/transformer/model.py``
(branch ``se_nca_h2h``, BM's architecture — see its ARCHITECTURE.md):

* **History encoder** — causal transformer over env steps; step token =
  CNN(o_t) ⊕ embed(a_t) ⊕ position. States b_0..b_{T-1}.
* **Frame-event decoder** — AR transformer over the current step's event
  tokens: self-attention causal *within the step block only*, cross-attention
  over b_{<=step}, logits over the event vocabulary, legality-masked so the
  model is an exactly normalized distribution over valid frame deltas.

Differences from the source: grid geometry (C, H, W), action count and the
event vocabulary are config values (script-doctor games vary per game;
N_ACTIONS = 6, no restart/RESET token); the planning-time cached paths
(encode_ext / decode_ctx / decode_step) are not ported yet — training and
teacher-forced eval only.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

NEG = -1e9


@dataclass
class EWMConfig:
    # data/geometry (per game)
    c_chan: int = 5
    h: int = 12
    w: int = 12
    t_len: int = 8                 # env steps per trajectory (L)
    max_len: int = 256             # decoder positions per trajectory
    num_actions: int = 6
    # encoder
    d_enc: int = 256
    enc_layers: int = 4
    enc_heads: int = 8
    # decoder
    d_dec: int = 192
    dec_layers: int = 3
    dec_heads: int = 6
    ffn_mult: int = 4
    conv_channels: tuple[int, ...] = (32, 64)
    max_step_tokens: int = 64      # within-step position table size
    # optimization
    lr: float = 3e-4
    warmup_steps: int = 1_000
    total_steps: int = 50_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 32
    seed: int = 0
    # run
    eval_every: int = 500
    ckpt_every: int = 0
    run_name: str = "event_wm"
    run_dir: str = ""
    wandb_project: str = ""
    wandb_entity: str = ""
    max_hours: float = 0.0

    @property
    def n_cell(self) -> int:
        return self.c_chan * self.h * self.w

    @property
    def vocab(self) -> int:
        return 2 * self.n_cell + 3

    @property
    def bos(self) -> int:
        return 2 * self.n_cell + 1

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.asdict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EWMConfig":
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        raw = {k: v for k, v in raw.items() if k in known}
        if "conv_channels" in raw:
            raw["conv_channels"] = tuple(raw["conv_channels"])
        return cls(**raw)


class _MHA(nn.Module):
    d: int
    heads: int

    def setup(self) -> None:
        hd = self.d // self.heads
        self.q_proj = nn.DenseGeneral((self.heads, hd), name="q")
        self.k_proj = nn.DenseGeneral((self.heads, hd), name="k")
        self.v_proj = nn.DenseGeneral((self.heads, hd), name="v")
        self.o_proj = nn.DenseGeneral(self.d, axis=(-2, -1), name="o")

    def __call__(self, q_in, kv_in, mask):
        """mask: [B, Lq, Lk] bool (True = attend)."""
        q, k, v = self.q_proj(q_in), self.k_proj(kv_in), self.v_proj(kv_in)
        hd = q.shape[-1]
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(hd)
        logits = jnp.where(mask[:, None], logits, NEG)
        att = jax.nn.softmax(logits, axis=-1)
        return self.o_proj(jnp.einsum("bhqk,bkhd->bqhd", att, v))


class _EncBlock(nn.Module):
    d: int
    heads: int
    ffn: int

    def setup(self) -> None:
        self.ln0 = nn.LayerNorm(name="LayerNorm_0")
        self.mha = _MHA(self.d, self.heads, name="_MHA_0")
        self.ln1 = nn.LayerNorm(name="LayerNorm_1")
        self.fc_out = nn.Dense(self.d, name="Dense_0")
        self.fc_in = nn.Dense(self.ffn, name="Dense_1")

    def __call__(self, x, mask):
        h = self.ln0(x)
        x = x + self.mha(h, h, mask)
        h = self.ln1(x)
        return x + self.fc_out(nn.gelu(self.fc_in(h)))


class _DecBlock(nn.Module):
    d: int
    heads: int
    ffn: int

    def setup(self) -> None:
        self.ln0 = nn.LayerNorm(name="LayerNorm_0")
        self.self_attn = _MHA(self.d, self.heads, name="self")
        self.ln1 = nn.LayerNorm(name="LayerNorm_1")
        self.cross_attn = _MHA(self.d, self.heads, name="cross")
        self.ln2 = nn.LayerNorm(name="LayerNorm_2")
        self.fc_out = nn.Dense(self.d, name="Dense_0")
        self.fc_in = nn.Dense(self.ffn, name="Dense_1")

    def __call__(self, x, enc, self_mask, cross_mask):
        h = self.ln0(x)
        x = x + self.self_attn(h, h, self_mask)
        h = self.ln1(x)
        x = x + self.cross_attn(h, enc, cross_mask)
        h = self.ln2(x)
        return x + self.fc_out(nn.gelu(self.fc_in(h)))


class EventWorldModel(nn.Module):
    cfg: EWMConfig

    def setup(self) -> None:
        c = self.cfg
        self.convs = [nn.Conv(ch, (3, 3), padding="SAME", name=f"conv{i}")
                      for i, ch in enumerate(c.conv_channels)]
        self.obs_proj = nn.Dense(c.d_enc, name="obs_proj")
        self.act_embed = nn.Embed(c.num_actions + 1, 64, name="act_embed")
        self.enc_pos = nn.Embed(c.t_len, c.d_enc, name="enc_pos")
        self.enc_in = nn.Dense(c.d_enc, name="enc_in")
        self.enc_blocks = [
            _EncBlock(c.d_enc, c.enc_heads, c.d_enc * c.ffn_mult,
                      name=f"enc{i}") for i in range(c.enc_layers)]
        self.enc_ln = nn.LayerNorm(name="enc_ln")
        self.tok_embed = nn.Embed(c.vocab, c.d_dec, name="tok_embed")
        self.step_embed = nn.Embed(c.t_len + 1, c.d_dec, name="step_embed")
        self.pos_embed = nn.Embed(c.max_step_tokens, c.d_dec,
                                  name="pos_embed")
        self.dec_blocks = [
            _DecBlock(c.d_dec, c.dec_heads, c.d_dec * c.ffn_mult,
                      name=f"dec{i}") for i in range(c.dec_layers)]
        self.dec_ln = nn.LayerNorm(name="dec_ln")
        self.head = nn.Dense(c.vocab, name="head")

    def encode(self, obs: jnp.ndarray, actions: jnp.ndarray,
               step_valid: jnp.ndarray) -> jnp.ndarray:
        """obs [B, >=T, C, H, W], actions [B, T], step_valid [B, T] bool
        (False = padding hole in the trajectory) -> [B, T, d_enc]."""
        c = self.cfg
        x = obs[:, :c.t_len].astype(jnp.float32)
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, c.c_chan, c.h, c.w).transpose(0, 2, 3, 1)
        for conv in self.convs:
            x = nn.relu(conv(x))
        x = self.obs_proj(x.reshape(b * t, -1)).reshape(b, t, -1)
        tok = self.enc_in(jnp.concatenate(
            [x, self.act_embed(actions)], axis=-1))
        tok = tok + self.enc_pos(jnp.arange(t))
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))[None]
        mask = causal & step_valid[:, None, :]
        mask = mask | jnp.eye(t, dtype=bool)[None]
        for blk in self.enc_blocks:
            tok = blk(tok, mask)
        return self.enc_ln(tok)

    def decode(self, enc: jnp.ndarray, dec_in: jnp.ndarray,
               dec_step: jnp.ndarray, dec_pos: jnp.ndarray,
               step_valid: jnp.ndarray) -> jnp.ndarray:
        """Teacher-forced decoder pass -> logits [B, L, vocab].

        ``dec_step`` uses anything >= t_len for padding; ``dec_pos`` is the
        within-step position (clipped to the table)."""
        c = self.cfg
        step = jnp.minimum(dec_step, c.t_len)         # pad bucket = t_len
        x = self.tok_embed(dec_in) + self.step_embed(step) \
            + self.pos_embed(jnp.minimum(dec_pos, c.max_step_tokens - 1))
        pad = dec_step >= c.t_len
        same = step[:, :, None] == step[:, None, :]
        causal = jnp.arange(x.shape[1])[None, :, None] >= \
            jnp.arange(x.shape[1])[None, None, :]
        self_mask = same & causal & ~pad[:, :, None] & ~pad[:, None, :]
        self_mask = self_mask | jnp.eye(x.shape[1], dtype=bool)[None]
        # cross-attention: encoder positions <= step, and valid
        cross_mask = (jnp.arange(enc.shape[1])[None, None, :] <=
                      step[:, :, None]) & step_valid[:, None, :]
        cross_mask = cross_mask | jax.nn.one_hot(
            jnp.minimum(step, enc.shape[1] - 1), enc.shape[1],
            dtype=bool)[:, :, :]
        for blk in self.dec_blocks:
            x = blk(x, enc, self_mask, cross_mask)
        return self.head(self.dec_ln(x))

    def __call__(self, obs, actions, dec_in, dec_step, dec_pos, step_valid):
        return self.decode(self.encode(obs, actions, step_valid),
                           dec_in, dec_step, dec_pos, step_valid)


def legality_mask(cfg: EWMConfig, obs: jnp.ndarray, dec_in: jnp.ndarray,
                  dec_step: jnp.ndarray) -> jnp.ndarray:
    """[B, L, vocab] bool: legal next tokens at each decoder position.

    Rules: frame consistency (remove needs occupied, add needs empty),
    strictly ascending event ids, EOF always, BOS/PAD never.
    """
    n = cfg.n_cell
    b, l = dec_in.shape
    step = jnp.minimum(dec_step, cfg.t_len - 1)
    frames = (obs[:, :cfg.t_len].reshape(b, cfg.t_len, n) > 0)
    cur = jnp.take_along_axis(frames, step[:, :, None], axis=1)  # [B,L,n]
    events_frame_ok = jnp.concatenate([cur, ~cur], axis=-1)      # [B,L,2n]
    prev_rank = jnp.where(dec_in == cfg.bos, -1, dec_in)
    ids = jnp.arange(2 * n)
    events_order_ok = ids[None, None, :] > prev_rank[:, :, None]
    events_ok = events_frame_ok & events_order_ok
    eof_ok = jnp.ones((b, l, 1), dtype=bool)
    no = jnp.zeros((b, l, 1), dtype=bool)
    return jnp.concatenate([events_ok, eof_ok, no, no], axis=-1)


def masked_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask, logits, NEG)


def dec_positions(dec_step: Any) -> Any:
    """Within-step position index (numpy, host side): 0 at each step start."""
    import numpy as np

    ds = np.asarray(dec_step)
    starts = np.ones_like(ds, dtype=bool)
    starts[:, 1:] = ds[:, 1:] != ds[:, :-1]
    idx = np.arange(ds.shape[1])[None, :].repeat(ds.shape[0], axis=0)
    start_idx = np.where(starts, idx, 0)
    start_idx = np.maximum.accumulate(start_idx, axis=1)
    return (idx - start_idx).astype(np.int32)
