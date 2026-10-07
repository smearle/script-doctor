"""Decoder-only transformer for PuzzleScript source (RoPE, RMSNorm, SwiGLU, tied embeddings)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    max_seq_len: int = 8192
    dropout: float = 0.0
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)


def rope_tables(seq_len: int, head_dim: int, base: float, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh); cos/sin: (T, Dh/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None, pos: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            kc, vc = cache
            kc[:, :, pos:pos + T] = k
            vc[:, :, pos:pos + T] = v
            k, v = kc[:, :, :pos + T], vc[:, :, :pos + T]
            # prefill from position 0 is causal; single-token decode attends to everything cached
            y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
            assert T == 1 or pos == 0, "multi-token prefill must start at position 0"
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w12 = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, pos: int = 0):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, cache, pos))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight
        cos, sin = rope_tables(cfg.max_seq_len, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - (self.wte.weight.numel() if non_embedding else 0)

    def forward(self, idx, caches=None, pos: int = 0):
        T = idx.shape[1]
        cos, sin = self.rope_cos[pos:pos + T], self.rope_sin[pos:pos + T]
        x = self.drop(self.wte(idx))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, None if caches is None else caches[i], pos)
        return self.head(self.ln_f(x))

    def new_caches(self, batch: int, length: int, dtype, device):
        hd = self.cfg.d_model // self.cfg.n_head
        shape = (batch, self.cfg.n_head, length, hd)
        return [(torch.zeros(shape, dtype=dtype, device=device),
                 torch.zeros(shape, dtype=dtype, device=device)) for _ in self.blocks]

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, eos_id: int, temperature: float = 1.0,
                 top_p: float = 1.0, generator: torch.Generator | None = None):
        """Sample continuations of equal-length prompts (B, T0). Returns list of new-token lists
        (each ends at, and excludes, the first eos) and a per-row flag for reaching eos."""
        B, T0 = prompt.shape
        total = min(self.cfg.max_seq_len, T0 + max_new)
        dtype = next(self.parameters()).dtype
        if prompt.is_cuda:
            dtype = torch.bfloat16
        caches = self.new_caches(B, total, dtype, prompt.device)
        out = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=prompt.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prompt.is_cuda):
            logits = self(prompt, caches, 0)[:, -1]
            pos = T0
            while pos < total:
                logits = logits.float() / max(temperature, 1e-6)
                if top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = sl.softmax(-1).cumsum(-1)
                    sl[cp - sl.softmax(-1) > top_p] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(-1, si, sl)
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
                nxt_l = nxt.squeeze(1).tolist()
                for b in range(B):
                    if not done[b]:
                        if nxt_l[b] == eos_id:
                            done[b] = True
                        else:
                            out[b].append(nxt_l[b])
                if bool(done.all()) or pos + 1 >= total:
                    break
                logits = self(nxt, caches, pos)[:, -1]
                pos += 1
        return out, done.tolist()
