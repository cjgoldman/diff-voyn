"""Per-language NELBO canary metric (task 0.6; full harness arrives in 3.1).

Estimates the Rao-Blackwellized continuous-time NELBO (design §1) in
bits/char under the log-linear schedule (α_t = 1−t):

    NELBO = ∫₀¹ (1/t) · E_q[ Σ_{i masked} −log p(x_i | z_t, L) ] dt

with stratified timestep sampling and an explicit seed so the same masking
realizations can be replayed across language conditions (common random
numbers — the variance-reduction lever the ranking depends on, design §5a).

Sanity anchor: a uniform random model scores log2(32) = 5 bits/char.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


@torch.no_grad()
def estimate_nelbo_bits_per_char(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ids: torch.Tensor,
    lang_idx: int,
    *,
    n_strata: int = 64,
    samples_per_stratum: int = 1,
    seed: int = 0,
    device: str | torch.device = "cpu",
    t_floor: float = 1e-3,
) -> float:
    """Estimate NELBO of ``ids`` [B, L] conditioned on ``lang_idx``.

    ``model(z_t, lang_idx_tensor) -> logits [B, L, V]``. The masking draws are
    a pure function of ``seed`` — score every candidate language with the same
    seed to get common random numbers.
    """
    from ..vocab import MASK_ID

    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = ids.to(device)
    batch, seq_len = ids.shape
    lang = torch.full((batch,), lang_idx, dtype=torch.long, device=device)

    total = 0.0
    n_terms = 0
    for s in range(n_strata):
        for _ in range(samples_per_stratum):
            u = torch.rand(1, generator=g).item()
            t = max((s + u) / n_strata, t_floor)
            mask_draw = torch.rand(batch, seq_len, generator=g).to(device)
            masked = mask_draw < t
            if not masked.any():
                n_terms += 1
                continue
            z_t = ids.masked_fill(masked, MASK_ID)
            logits = model(z_t, lang)
            ce = F.cross_entropy(
                logits[masked], ids[masked], reduction="sum"
            )  # nats over masked positions
            total += (1.0 / t) * ce.item() / (batch * seq_len)
            n_terms += 1

    nats_per_char = total / n_terms
    return nats_per_char / math.log(2.0)


@torch.no_grad()
def per_window_nelbo_bits(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ids: torch.Tensor,
    lang_idx: int,
    *,
    n_strata: int = 16,
    seed: int = 0,
    device: str | torch.device = "cpu",
    t_floor: float = 1e-3,
) -> torch.Tensor:
    """Per-window NELBO estimates [B] in bits/char (task 3.3 groundwork).

    Same stratified estimator and CRN semantics as
    :func:`estimate_nelbo_bits_per_char` (identical ``seed`` ⇒ identical t
    values and mask positions), but returns one value per window instead of
    the batch mean — the primitive behind per-document mean-and-spread
    scoring and the 25M/85M ranking-agreement probe (task 1.6).
    """
    from ..vocab import MASK_ID

    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = ids.to(device)
    batch, seq_len = ids.shape
    lang = torch.full((batch,), lang_idx, dtype=torch.long, device=device)

    total = torch.zeros(batch, dtype=torch.float64)
    for s in range(n_strata):
        u = torch.rand(1, generator=g).item()
        t = max((s + u) / n_strata, t_floor)
        mask_draw = torch.rand(batch, seq_len, generator=g).to(device)
        masked = mask_draw < t
        if not masked.any():
            continue
        z_t = ids.masked_fill(masked, MASK_ID)
        logp = F.log_softmax(model(z_t, lang).float(), dim=-1)
        nll = -logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        nll = nll.masked_fill(~masked, 0.0)
        total += (nll.sum(dim=-1) / (t * seq_len)).double().cpu()

    return (total / n_strata / math.log(2.0)).float()
