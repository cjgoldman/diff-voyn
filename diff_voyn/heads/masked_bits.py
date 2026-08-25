"""Partially-observed (blanked) scoring of a decode — the "confidence mask"
probe.

The judge is an absorbing-state masked diffusion model, so "blank" is a token
it was trained to reconstruct from: a position we do not trust can be handed
to it as a permanent ``MASK`` instead of a possibly-wrong letter. This module
scores such a partially observed stream:

- every *unobserved* position sits at ``MASK`` in ``z_t`` for every stratum
  (it still shapes the context, it is never a prediction target);
- the loss sums only over positions that are BOTH observed and masked by the
  stratum draw;
- the result is normalized per **observed** character, so a decode and its
  shuffled control (same mask) are on the same scale.

With ``observed`` all-true this reproduces :func:`heads.two_tier.paired_bits`
bit-for-bit (identical draw order: per stratum ``u`` then one ``(1, L)``
mask), which is what ``tests/test_masked_bits.py`` asserts — the baseline arm
of the experiment is therefore the frozen Phase-6 estimator, not a re-
implementation of it.

Caveat on interpretation: the returned number is a NELBO-style upper bound on
the observed subsequence's description length *given the blanks as context*,
not the Phase-6 bits/char. It is comparable across (decode, shuffled) pairs
and across language conditions that share a mask; it is NOT comparable across
different masks, because the conditioning set changes with the mask.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..vocab import LETTER_IDS, MASK_ID
from .two_tier import condition_index

LETTER_BASE = LETTER_IDS[0]


@torch.no_grad()
def paired_bits_masked(
    evaluator,
    rows: np.ndarray,
    observed: np.ndarray,
    conditions: list[str],
    *,
    n_strata: int = 64,
    seed: int = 0,
    batch: int = 32,
) -> np.ndarray:
    """``[n, C]`` bits per **observed** char of ``rows`` ``[n, L]`` (letter
    ids 0..A-1) under each condition, with one masking realization shared by
    all rows and all conditions (CRN, non-negotiable #4).

    ``observed`` is ``[n, L]`` bool (or ``[L]``, broadcast to every row).
    Rows longer than the evaluator window are scored in windows and averaged
    weighted by each window's *observed* count.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n, L = rows.shape
    obs = np.asarray(observed, dtype=bool)
    if obs.ndim == 1:
        obs = np.broadcast_to(obs, (n, L))
    if obs.shape != (n, L):
        raise ValueError(f"observed {obs.shape} does not match rows {(n, L)}")
    if not obs.any(axis=1).all():
        raise ValueError("every row needs at least one observed position")
    cuts = evaluator._windows(L)
    if len(cuts) > 1:
        cuts = [(a, b) for a, b in cuts if b - a >= evaluator.window // 2]
    total = np.zeros((n, len(conditions)))
    weight = np.zeros(n)
    for wi, (a, b) in enumerate(cuts):
        w_obs = obs[:, a:b]
        if not w_obs.any(axis=1).all():
            continue  # a window with a fully blank row contributes nothing
        ids = torch.from_numpy(rows[:, a:b] + LETTER_BASE).to(evaluator.device)
        o = torch.from_numpy(np.ascontiguousarray(w_obs)).to(evaluator.device)
        cnt = w_obs.sum(axis=1).astype(float)
        total += cnt[:, None] * _paired_window_masked(
            evaluator, ids, o, conditions, n_strata, seed + 7919 * wi, batch
        )
        weight += cnt
    return total / weight[:, None]


def _paired_window_masked(evaluator, ids, obs, conditions, n_strata, seed, batch):
    """One window. Draw order per stratum is ``u`` then one ``(1, L)`` mask —
    identical to :func:`heads.two_tier._paired_window`, so an all-observed
    call replays exactly the same masking realizations."""
    n, L = ids.shape
    dev = evaluator.device
    model = evaluator.backbone
    g = torch.Generator(device="cpu").manual_seed(seed)
    ts, masks = [], []
    for s in range(n_strata):
        u = torch.rand(1, generator=g).item()
        t = max((s + u) / n_strata, evaluator.t_floor)
        m = torch.rand(1, L, generator=g) < t
        if m.any():
            ts.append(t)
            masks.append(m[0])
    masks = torch.stack(masks).to(dev)  # (K, L)
    t_vec = torch.tensor(ts, dtype=torch.float64)
    K = masks.shape[0]
    n_obs = obs.sum(-1).double().cpu()  # (n,)
    # A blank is MASK in the input for EVERY stratum; a target only where the
    # stratum masked it AND we observe it.
    blanked = masks[:, None, :] | (~obs)[None]  # (K, n, L)
    z_all = torch.where(blanked, torch.full_like(ids, MASK_ID)[None], ids[None])
    tgt_mask = masks[:, None, :] & obs[None]  # (K, n, L)
    z_all = z_all.reshape(K * n, L)
    tgt = ids[None].expand(K, n, L).reshape(K * n, L)
    m_all = tgt_mask.reshape(K * n, L)
    acc = torch.zeros(n, len(conditions), dtype=torch.float64)
    for j, cond in enumerate(conditions):
        lang_idx = condition_index(cond)
        out = torch.zeros(K * n, dtype=torch.float64)
        for start in range(0, K * n, batch):
            zb = z_all[start : start + batch]
            lang = torch.full((zb.shape[0],), lang_idx, dtype=torch.long, device=dev)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=evaluator.autocast
            ):
                logits = model(zb, lang)
            logp = F.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, tgt[start : start + batch].unsqueeze(-1)).squeeze(-1)
            nll = nll.masked_fill(~m_all[start : start + batch], 0.0)
            out[start : start + batch] = nll.sum(-1).double().cpu()
        acc[:, j] = (out.reshape(K, n) / (t_vec[:, None] * n_obs[None, :])).sum(0)
    return (acc / n_strata / math.log(2.0)).numpy()
