"""MDLM diffusion core — task 1.1 (design §1).

Continuous-time masked (absorbing-state) discrete diffusion:

- **Forward process** under the log-linear schedule ``α_t = 1 − t``: at time
  ``t ∈ (0, 1]`` each position of ``x`` is independently replaced by MASK with
  probability ``1 − α_t = t``. This is exactly what
  :class:`diff_voyn.data.loader.MaskingSampler` draws.
- **SUBS parameterization**: the denoiser predicts a distribution with *zero
  masking probability* (the MASK logit is −inf — applied inside
  :class:`diff_voyn.model.backbone.Backbone`), and unmasked positions carry
  over their input token with probability one, so their loss terms vanish
  identically. Under SUBS the NELBO is invariant to schedule
  reparameterization and the network needs no time conditioning.
- **Rao-Blackwellized NELBO**::

      L = E_{t~U(0,1]} E_{z_t~q(·|x,t)} [ (α'_t / (1−α_t)) · Σ_{i: z_t,i=MASK}
                                          log p_θ(x_i | z_t, L) ]

  which with ``α_t = 1 − t`` gives the per-draw integrand
  ``(1/t) · Σ_{masked} −log p_θ(x_i | z_t, L)`` in nats. Losses here are
  normalized *per character of the full window* (divide by seq_len, not by
  the masked count), so the training loss is literally an upper bound on
  nats/char and comparable across masking draws and batch shapes.

Verified against an exact (enumerated + quadrature) reference and against the
discrete-time T-step MDLM bound in ``tests/test_diffusion.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..vocab import MASK_ID

LN2 = math.log(2.0)


def subs_parameterize(logits: torch.Tensor) -> torch.Tensor:
    """Zero the masking probability (SUBS): MASK logit → −inf, in place.

    Safe under autograd (the gradient through the overwritten entry is zero,
    which is exactly the parameterization: the model never predicts MASK).
    """
    logits[..., MASK_ID] = float("-inf")
    return logits


def mdlm_nelbo_terms(
    logits: torch.Tensor,
    ids: torch.Tensor,
    masked: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Per-example Rao-Blackwellized NELBO integrand, nats per character.

    ``logits`` [B, L, V] (SUBS-parameterized), ``ids`` [B, L] clean targets,
    ``masked`` [B, L] bool (the positions that were MASK in ``z_t``),
    ``t`` [B] the per-example diffusion times. Returns [B].

    An example with no masked position contributes 0 — an unbiased draw of an
    integrand that vanishes as t → 0, not a degenerate case.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # [B, L]
    nll = nll.masked_fill(~masked, 0.0)
    return nll.sum(dim=-1) / (t * ids.shape[-1])


def mdlm_loss(
    logits: torch.Tensor,
    ids: torch.Tensor,
    masked: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Batch-mean NELBO in nats/char (divide by :data:`LN2` for bits/char)."""
    return mdlm_nelbo_terms(logits, ids, masked, t).mean()
