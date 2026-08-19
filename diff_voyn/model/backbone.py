"""Encoder-only diffusion backbone — task 1.2 (design §3).

ALICE-recipe transformer: bidirectional encoder, RMSNorm (pre-norm), SwiGLU
FFN, RoPE, no biases, no time-conditioning network (SUBS makes the objective
schedule-invariant, design §1). Two presets (``model_preset`` in
``diff_voyn.infra.config``):

- ``85m`` — 12 layers / d_model 768 / 12 heads / ffn 2048 (the instrument);
- ``25m`` — 6 layers / d_model 512 / 8 heads / ffn 1408 (restart-heavy search).

Language conditioning is the Phase-0 :class:`~diff_voyn.data.loader.
LanguageConditioning`: an additive per-position language embedding with 10%
conditioning dropout to a learned NULL-language embedding (design §4).

The forward output is SUBS-parameterized (MASK logit −inf), so every consumer
— training loss, NELBO canary, Phase-3 scoring harness — sees the same
zero-masking-probability distribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..data.loader import LanguageConditioning
from ..infra.config import ModelConfig
from ..vocab import VOCAB_SIZE
from .diffusion import subs_parameterize


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    """Precomputed RoPE tables, shared across layers."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(max_seq_len).float()[:, None] * inv_freq[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` [B, H, L, hd] by position."""
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len].to(x.dtype)  # [L, hd/2]
        sin = self.sin[:seq_len].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.attn_dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, rotary: Rotary) -> torch.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, l, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        q, k = rotary.rotate(q), rotary.rotate(k)
        # Bidirectional (no causal mask) — the model is an encoder by design.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout_p if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(b, l, d)
        return self.dropout(self.out(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.down = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rotary: Rotary) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rotary)
        return x + self.ffn(self.ffn_norm(x))


class Backbone(nn.Module):
    """``forward(z_t, lang_idx) -> SUBS logits [B, L, VOCAB_SIZE]``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.lang_cond = LanguageConditioning(cfg.d_model, cfg.lang_cond_dropout)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.rotary = Rotary(cfg.d_model // cfg.n_heads, cfg.seq_len)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.apply(self._init)
        # Scale residual-branch outputs down with depth (GPT-2 convention).
        scale = (2 * cfg.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 * scale)
            nn.init.normal_(block.ffn.down.weight, std=0.02 * scale)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        lang_idx: torch.Tensor,
        g: torch.Generator | None = None,
    ) -> torch.Tensor:
        h = self.embed(z_t) + self.lang_cond(lang_idx, g)
        h = self.embed_dropout(h)
        for block in self.blocks:
            h = block(h, self.rotary)
        return subs_parameterize(self.head(self.final_norm(h)))

    def forward_soft(
        self,
        probs: torch.Tensor,
        lang_idx: torch.Tensor,
        g: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mixture-input path (task 5.1 / design §8, R3): ``probs``
        [B, L, VOCAB_SIZE] row-stochastic distributions are fed as expected
        embeddings ``probs @ E``. With one-hot rows this reproduces
        :meth:`forward` exactly; soft rows carry dense gradients to a cipher
        head. No parameters differ from the id path."""
        h = probs @ self.embed.weight + self.lang_cond(lang_idx, g)
        h = self.embed_dropout(h)
        for block in self.blocks:
            h = block(h, self.rotary)
        return subs_parameterize(self.head(self.final_norm(h)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def param_count_str(model: Backbone) -> str:
    n = model.n_params()
    return f"{n / 1e6:.1f}M ({n:,})"


def flops_per_char(cfg: ModelConfig) -> float:
    """Rough forward FLOPs per character (2·N_params + attention term)."""
    n_dense = 12 * cfg.n_layers * cfg.d_model**2  # qkv+o (4d²) + swiglu (~8d²ish)
    attn = 2 * cfg.n_layers * cfg.seq_len * cfg.d_model
    return 2.0 * (n_dense + attn)


def language_dropout_rate(model: Backbone) -> float:
    """Realized conditioning-dropout fraction of the most recent training
    forward (task 1.2 acceptance: 'dropout rate verified in logs')."""
    return getattr(model.lang_cond, "last_drop_fraction", math.nan)
