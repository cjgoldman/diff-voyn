"""Per-language character AR reference model — task 3.4 (design §5b.3).

The diffusion NELBO is an upper bound on −log p(x); how loose that bound is
may differ by language, and the ranking compares per-language bounds (R1).
Design §5b.3: "where an AR reference is affordable, train a small character
AR transformer on the same data and use NELBO − NLL_AR per language as a
direct bound-gap estimate". This module is that reference model.

It reuses the backbone's blocks (RMSNorm / SwiGLU / RoPE, same vocab, same
1024-char context) with a causal mask, so the only modeling difference to
the instrument is autoregressive factorization — the comparison isolates
bound looseness from architecture choices as far as a small model can.
Each language gets its own model trained on exactly the backbone's train
split for that language (``CorpusWindows`` over the same doc ids).

Multilingual variant (calibration v3, task 3.4/3.5): the per-language
references are data-limited unequally (Italian has 3.6M train chars, German
89M), so ``NELBO − NLL_AR`` then mixes the bound gap with "how much the
reference was starved" — the confound the fairness audit flagged in v1. With
``ARConfig.multilingual=True`` one model carries the backbone's own additive
:class:`~diff_voyn.data.loader.LanguageConditioning` (same 10% conditioning
dropout) and is trained on the same τ-balanced three-language mix, so the
reference enjoys the same cross-lingual transfer as the instrument and the
only remaining difference is the factorization (+ capacity, which is
matched exactly for the 25M sibling).

Scoring convention — every character of a window is predicted, the first
from BOS, so ``nll_bits_per_char`` is over the same L characters the
diffusion NELBO averages over (``per_window_nelbo_bits``). A uniform model
scores exactly log2(32) = 5 bits/char, like the diffusion anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..data.loader import LanguageConditioning
from ..infra.config import ModelConfig
from ..vocab import BOS_ID, VOCAB_SIZE
from .backbone import Backbone, Block, RMSNorm, Rotary


@dataclass
class ARConfig:
    """Small by design (§5b.3 "small char AR transformer"): ~10M params so the
    3.6M-char Italian train split is not hopelessly overfit."""

    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ffn: int = 1024
    dropout: float = 0.1
    seq_len: int = 1024
    multilingual: bool = False  # language-conditioned single model (AR v3)
    lang_cond_dropout: float = 0.1  # only used when multilingual

    def as_model_config(self) -> ModelConfig:
        return ModelConfig(
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
            seq_len=self.seq_len + 1,  # RoPE table covers the BOS-shifted input
            lang_cond_dropout=self.lang_cond_dropout if self.multilingual else 0.0,
        )


def ar_preset(name: str = "10m") -> ARConfig:
    presets = {
        "10m": ARConfig(),
        "3m": ARConfig(n_layers=4, d_model=256, n_heads=4, d_ffn=704),
        # the 25M diffusion sibling's dims (infra.config.model_preset("25m")),
        # causal — capacity-matched reference for the multilingual variant
        "25m": ARConfig(n_layers=6, d_model=512, n_heads=8, d_ffn=1408),
    }
    return presets[name]


class CharARLM(nn.Module):
    """``forward(ids [B, L], lang_idx [B] | None) -> next-char logits
    [B, L, VOCAB_SIZE]`` where ``logits[:, i]`` predicts ``ids[:, i]`` from
    BOS + ``ids[:, :i]``. ``lang_idx`` is required iff ``cfg.multilingual``."""

    def __init__(self, cfg: ARConfig):
        super().__init__()
        mc = cfg.as_model_config()
        if mc.d_model % mc.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB_SIZE, mc.d_model)
        self.lang_cond = (
            LanguageConditioning(mc.d_model, mc.lang_cond_dropout)
            if cfg.multilingual
            else None
        )
        self.embed_dropout = nn.Dropout(mc.dropout)
        self.rotary = Rotary(mc.d_model // mc.n_heads, mc.seq_len)
        self.blocks = nn.ModuleList(Block(mc, causal=True) for _ in range(mc.n_layers))
        self.final_norm = RMSNorm(mc.d_model)
        self.head = nn.Linear(mc.d_model, VOCAB_SIZE, bias=False)
        self.apply(Backbone._init)
        scale = (2 * mc.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 * scale)
            nn.init.normal_(block.ffn.down.weight, std=0.02 * scale)

    def forward(
        self, ids: torch.Tensor, lang_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, l = ids.shape
        if l > self.cfg.seq_len:
            raise ValueError(f"window of {l} chars exceeds seq_len {self.cfg.seq_len}")
        bos = torch.full((b, 1), BOS_ID, dtype=ids.dtype, device=ids.device)
        x = torch.cat([bos, ids[:, :-1]], dim=1)
        h = self.embed(x)
        if self.lang_cond is not None:
            if lang_idx is None:
                raise ValueError("multilingual AR reference needs lang_idx")
            h = h + self.lang_cond(lang_idx)
        # monolingual model: conditioning is implicit, lang_idx is ignored
        h = self.embed_dropout(h)
        for block in self.blocks:
            h = block(h, self.rotary)
        return self.head(self.final_norm(h))

    def nll_nats(
        self, ids: torch.Tensor, lang_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-window summed −log p(x) in nats, [B]."""
        logits = self(ids, lang_idx).float()
        nll = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), ids.reshape(-1), reduction="none"
        )
        return nll.view(ids.shape).sum(dim=1)

    @torch.no_grad()
    def nll_bits_per_char(
        self, ids: torch.Tensor, lang_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-window bits/char, [B] — the §5b.3 ``NLL_AR`` term."""
        return self.nll_nats(ids, lang_idx) / (ids.shape[1] * math.log(2.0))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ar_loss(
    model: CharARLM, ids: torch.Tensor, lang_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean next-char cross-entropy in nats/char (training objective)."""
    return model.nll_nats(ids, lang_idx).sum() / ids.numel()
