"""Two-tier search — design §7.4: the cheap frozen n-gram DP runs the
restart-heavy inner search; the frozen diffusion ELBO decides.

``Candidate`` is what an inner search hands up: a hard decode plus the key
that produced it and the inner objective it reached. ``rescore`` puts every
candidate of one instance on the diffusion scale with PAIRED masking
realizations (identical t draws and mask positions for every candidate and
every language condition — the CRN discipline of non-negotiable #4 applied
inside a shortlist, where the decision is a difference between near-equal
texts). ``select`` picks the ELBO winner; the n-gram winner and the oracle
(best-SER) candidate are kept alongside so the delta the prototyping doc
§9 asks for is measured, not assumed.

Soft refinement (expected-embedding gradients through the frozen backbone,
R3) is head-specific and lives with each head (``refine_*`` functions in
the rung modules / scripts); it returns a new ``Candidate`` tagged
``source="refined"`` that enters the same shortlist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.loader import LANG_TO_INDEX
from ..vocab import LETTER_IDS, MASK_ID

LETTER_BASE = LETTER_IDS[0]


@dataclass
class Candidate:
    decode: np.ndarray  # hard plaintext letter ids 0..A-1
    key: Any  # head-specific key (perm / symbol map / block maps / (v, u))
    inner_score: float  # the inner search objective (higher is better)
    source: str = "search"
    bits: dict[str, float] = field(default_factory=dict)  # condition -> bits/char
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "inner_score": float(self.inner_score),
            "bits": dict(self.bits),
            "n_plain": len(self.decode),
            **{k: v for k, v in self.extra.items() if _jsonable(v)},
        }


def _jsonable(v) -> bool:
    return isinstance(v, (int, float, str, bool, list, dict, type(None)))


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    """Drop candidates with identical decodes (keep the first / best)."""
    seen, out = set(), []
    for c in cands:
        h = c.decode.tobytes()
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out


@torch.no_grad()
def paired_bits(
    evaluator,
    rows: np.ndarray,
    conditions: list[str],
    *,
    n_strata: int = 64,
    seed: int = 0,
    batch: int = 32,
) -> np.ndarray:
    """[n, C] bits/char of ``rows`` [n, L] (letter ids 0..A-1) under each
    condition with ONE masking realization shared by all rows and all
    conditions. Same estimator as ``metrology.per_window_nelbo_bits`` (for a
    single row and the same seed the draw sequence is identical, so a
    candidate scored here equals its metrology score). Rows longer than the
    evaluator window are scored in windows and length-averaged."""
    rows = np.asarray(rows, dtype=np.int64)
    n, L = rows.shape
    cuts = evaluator._windows(L)
    if len(cuts) > 1:
        cuts = [(a, b) for a, b in cuts if b - a >= evaluator.window // 2]
    total = np.zeros((n, len(conditions)))
    weight = 0.0
    for wi, (a, b) in enumerate(cuts):
        ids = torch.from_numpy(rows[:, a:b] + LETTER_BASE).to(evaluator.device)
        total += (b - a) * _paired_window(
            evaluator, ids, conditions, n_strata, seed + 7919 * wi, batch
        )
        weight += b - a
    return total / weight


def _paired_window(evaluator, ids, conditions, n_strata, seed, batch):
    """Strata × rows flattened into batched forwards (draw order per stratum:
    u, then one (1, L) mask — identical to the metrology estimator)."""
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
    acc = torch.zeros(n, len(conditions), dtype=torch.float64)
    # rows = (stratum k, row i) pairs
    z_all = torch.where(
        masks[:, None, :], torch.full_like(ids, MASK_ID)[None], ids[None]
    )  # (K, n, L)
    z_all = z_all.reshape(K * n, L)
    tgt = ids[None].expand(K, n, L).reshape(K * n, L)
    m_all = masks[:, None, :].expand(K, n, L).reshape(K * n, L)
    for j, cond in enumerate(conditions):
        lang_idx = LANG_TO_INDEX[cond]
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
        acc[:, j] = (out.reshape(K, n) / (t_vec[:, None] * L)).sum(0)
    return (acc / n_strata / math.log(2.0)).numpy()


def rescore(
    evaluator,
    cands: list[Candidate],
    *,
    language: str,
    conditions: list[str] | None = None,
    n_strata: int = 64,
    seed: int = 0,
    batch: int = 32,
) -> list[Candidate]:
    """Fill ``c.bits[condition]`` for every candidate (paired masks). Decodes
    of unequal length (rung 4) are grouped by length; groups share the seed
    so equal-length decodes are still paired."""
    conds = list(conditions or [language])
    by_len: dict[int, list[Candidate]] = {}
    for c in cands:
        by_len.setdefault(len(c.decode), []).append(c)
    for group in by_len.values():
        rows = np.stack([c.decode for c in group])
        bits = paired_bits(
            evaluator, rows, conds, n_strata=n_strata, seed=seed, batch=batch
        )
        for c, b in zip(group, bits):
            for j, cond in enumerate(conds):
                c.bits[cond] = float(b[j])
    return cands


def select(cands: list[Candidate], *, language: str) -> dict[str, Candidate]:
    """The three decisions of the delta protocol: inner-search winner,
    diffusion winner (lowest own-condition bits), and — when ground truth is
    attached as ``extra['ser']`` — the oracle."""
    out = {
        "ngram": max(cands, key=lambda c: c.inner_score),
        "diffusion": min(cands, key=lambda c: c.bits[language]),
    }
    with_truth = [c for c in cands if "ser" in c.extra]
    if with_truth:
        out["oracle"] = min(with_truth, key=lambda c: c.extra["ser"])
    return out
