"""
A small decoder-only transformer for next-token prediction on TV-world traces.

Vocab is tiny (~37 tokens), so the model can be very small. The architecture
is a standard pre-norm transformer with rotary-free learned positional
embeddings and tied input/output projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_freqs(d_head: int, max_seq_len: int, base: float = 10000.0
                ) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    theta = 1.0 / (base ** (i / d_head))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, theta)          # (max_seq_len, d_head/2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D)  cos/sin: (max_seq_len, D/2) — sliced to T internally."""
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)    # (1, 1, T, D/2)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


@dataclass
class ModelConfig:
    vocab_size: int = 64
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.0
    pad_id: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout
        cos, sin = _rope_freqs(self.d_head, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)                  # (B, T, H, D)
        q = _apply_rope(q.transpose(1, 2), self.rope_cos, self.rope_sin)  # (B, H, T, D)
        k = _apply_rope(k.transpose(1, 2), self.rope_cos, self.rope_sin)
        v = v.transpose(1, 2)

        if key_pad_mask is not None:
            # Combine causal mask with padding mask into one additive mask so
            # we don't pass both attn_mask and is_causal=True (disallowed on MPS).
            causal = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
            )  # (T, T)
            pad = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype).masked_fill(
                ~key_pad_mask[:, None, None, :], float("-inf")
            )  # (B, 1, 1, T)
            attn_mask = causal[None, None, :, :] + pad  # (B, 1, T, T)
            is_causal = False
        else:
            attn_mask = None
            is_causal = True

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Tied weights: lm_head shares weights with the input embedding.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self,
                input_ids: torch.Tensor,
                key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (T, self.cfg.max_seq_len)
        x = self.tok(input_ids)
        for blk in self.blocks:
            x = blk(x, key_pad_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def num_params(self) -> int:
        # Don't double-count tied weights.
        seen = set()
        n = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n += p.numel()
        return n


def causal_lm_loss(logits: torch.Tensor,
                   input_ids: torch.Tensor,
                   loss_mask: torch.Tensor,
                   pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard next-token cross-entropy.

    logits: (B, T, V)
    input_ids: (B, T)
    loss_mask: (B, T)   True at real (non-pad) positions
    Returns (loss, num_tokens).
    """
    # Predict token t+1 from positions <= t.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    V = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        reduction="none",
        ignore_index=pad_id,
    )
    mask_flat = shift_mask.view(-1)
    n = mask_flat.sum().clamp_min(1)
    loss = (loss_flat * mask_flat.float()).sum() / n
    return loss, n


if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=37, d_model=64, n_layer=2, n_head=4, d_ff=128, max_seq_len=64)
    m = TinyTransformerLM(cfg)
    x = torch.randint(1, 37, (2, 16))
    mask = torch.ones_like(x, dtype=torch.bool)
    logits = m(x, mask)
    print("logits:", logits.shape)
    print("params:", m.num_params())
