"""Denoiser posteriors as *proposals* for the n-gram key search
(``docs/alt_loop_plan.md`` §2, mechanism M1).

The frozen diffusion backbone predicts, at every masked slot, a
distribution over the vocabulary given the unmasked context. Decoding a
ciphertext under the current key, masking a fraction of the positions and
reading those predictions gives a per-position letter posterior that
conditions on ~1000 characters of context — information a local n-gram
objective does not have. Summed over the occurrences of a cipher symbol
it is a score for every reassignment of that symbol, from which the
symbols the judge *disagrees* with the key on can be ranked. Nothing here
scores a key: that stays with ``two_tier.paired_bits`` / ``race_polish``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..vocab import LETTER_IDS, MASK_ID, VOCAB_SIZE
from .frame import letters_to_vocab

A = len(LETTER_IDS)


@torch.no_grad()
def position_posterior(
    evaluator,
    letters: np.ndarray,
    language: str,
    *,
    n_draws: int = 16,
    mask_rate: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """(L, A) mean log-posterior over letters at every position of the hard
    letter stream ``letters`` (ids 0..A-1), averaged over ``n_draws``
    independent masks at rate ``mask_rate``; positions never masked get a
    uniform row. Windows longer than the evaluator's context are scored in
    the evaluator's own tiling."""
    from .two_tier import condition_index

    ids = np.asarray(letters, dtype=np.int64)
    L = len(ids)
    out = np.zeros((L, A))
    cnt = np.zeros(L)
    g = torch.Generator().manual_seed(seed)
    onehot = torch.zeros(L, A)
    onehot[torch.arange(L), torch.from_numpy(ids)] = 1.0
    frame = letters_to_vocab(onehot).to(evaluator.device)
    mask_onehot = torch.zeros(VOCAB_SIZE, device=evaluator.device)
    mask_onehot[MASK_ID] = 1.0
    letter_ids = torch.tensor(LETTER_IDS, device=evaluator.device)
    bs = getattr(evaluator, "stratum_batch", 16)
    for a, b in evaluator._windows(L):
        S = b - a
        masks = torch.rand(n_draws, S, generator=g) < mask_rate
        masks = masks.to(evaluator.device)
        for i in range(0, n_draws, bs):
            m = masks[i : i + bs]
            B = m.shape[0]
            z = torch.where(m[:, :, None], mask_onehot[None, None, :], frame[a:b][None])
            lang = torch.full((B,), condition_index(language), device=evaluator.device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=evaluator.autocast
            ):
                logits = evaluator.backbone.forward_soft(z, lang)
            logq = torch.log_softmax(
                logits.float()[:, :, letter_ids], dim=-1
            )  # (B,S,A)
            logq = logq * m[:, :, None]
            out[a:b] += logq.sum(0).cpu().numpy()
            cnt[a:b] += m.sum(0).cpu().numpy()
    seen = cnt > 0
    out[seen] /= cnt[seen][:, None]
    out[~seen] = -np.log(A)
    return out


def symbol_scores(
    pos_logp: np.ndarray, cipher_ids: np.ndarray, n_symbols: int
) -> np.ndarray:
    """(n_symbols, A) summed per-position log-posterior over each symbol's
    occurrences: ``scores[s, a]`` is the judge's support for mapping ``s``
    to ``a`` (occurrence-weighted by construction)."""
    cipher = np.asarray(cipher_ids, dtype=np.int64)
    scores = np.zeros((n_symbols, pos_logp.shape[1]))
    np.add.at(scores, cipher, pos_logp)
    return scores


def unit_scores(
    pos_logp: np.ndarray,
    symbols: np.ndarray,
    sym_map: np.ndarray,
    targets,
) -> np.ndarray:
    """Word-homophonic variant: ``(n_types, n_units)`` support for mapping a
    type to each unit of the *same length class* as its current unit
    (letter types over the letters, bigram types over the bigrams);
    other entries are ``-inf``. Positions follow ``expand_units`` of the
    current decode."""
    symbols = np.asarray(symbols, dtype=np.int64)
    units = sym_map[symbols]
    n_types = len(sym_map)
    n_units = targets.n
    scores = np.full((n_types, n_units), -np.inf)
    big = np.asarray(targets.bigrams, dtype=np.int64).reshape(-1, 2)
    letter_of_unit = np.arange(A)
    single = np.zeros((n_types, A))
    pair = np.zeros((n_types, len(big)))
    pos = 0
    for tok, u in zip(symbols, units):
        if u < A:
            single[tok] += pos_logp[pos]
            pos += 1
        else:
            if len(big):
                pair[tok] += pos_logp[pos, big[:, 0]] + pos_logp[pos + 1, big[:, 1]]
            pos += 2
    assert pos == pos_logp.shape[0], (pos, pos_logp.shape)
    is_letter = sym_map < A
    scores[is_letter, :A] = single[is_letter][:, letter_of_unit]
    if len(big):
        scores[~is_letter, A:] = pair[~is_letter]
    return scores


def disagreements(
    scores: np.ndarray, sym_map: np.ndarray
) -> list[tuple[int, int, float]]:
    """Symbols whose best-supported unit differs from the key, as
    ``(symbol, proposed_unit, margin)`` sorted by margin (summed log-posterior
    gain, i.e. occurrence-weighted) descending."""
    sym_map = np.asarray(sym_map, dtype=np.int64)
    best = scores.argmax(1)
    cur = scores[np.arange(len(sym_map)), sym_map]
    margin = scores[np.arange(len(sym_map)), best] - cur
    out = [
        (int(s), int(best[s]), float(margin[s]))
        for s in range(len(sym_map))
        if best[s] != sym_map[s] and np.isfinite(margin[s]) and margin[s] > 0
    ]
    out.sort(key=lambda x: -x[2])
    return out
