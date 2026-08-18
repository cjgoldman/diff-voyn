"""G0 plumbing stub — a random-init denoiser with the real conditioning
interface, used only to verify that the per-language NELBO metric pipeline
logs to ClearML before Phase 1 launches (gate G0). NOT the Phase-1 backbone.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.loader import LanguageConditioning
from ..vocab import VOCAB_SIZE


class StubDenoiser(nn.Module):
    """Embedding + additive language conditioning + 2 vanilla encoder layers."""

    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.lang_cond = LanguageConditioning(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, z_t: torch.Tensor, lang_idx: torch.Tensor) -> torch.Tensor:
        h = self.embed(z_t) + self.lang_cond(lang_idx)
        return self.head(self.encoder(h))
