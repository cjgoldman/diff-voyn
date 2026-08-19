"""2N-slot expected-embedding frame — task 5.1 / CH.4 (design §8).

Each ciphertext token owns TWO plaintext slots. Slot 1 carries the token's
first (or only) letter distribution; slot 2 is a log-space blend of the
token's second-letter distribution and the learned ``NULL`` symbol, weighted
by the token's unigram-vs-bigram weight ``w_t`` (w_t = P(token is unigram) →
slot 2 is NULL with probability w_t).

The known numerical trap (design §8, non-negotiable #6): a log-space blend
``logaddexp(log w + log p_null, log(1-w) + log p_letter)`` hits
``logaddexp(-inf, -inf)`` wherever BOTH operands vanish — e.g. a vocab entry
that is neither a real letter nor NULL, with w exactly 0 or 1. The guard:
blend the finite real-letter block and the NULL entry separately in
probability space per vocab slot, never summing two -inf logs. We therefore
blend in PROBABILITY space (the inputs are row-stochastic, exactly
representable) and only take logs where a consumer needs them.

Heads emit this frame from day one even though only the diffusion evaluator
consumes it — that is what makes the post-G4 swap mechanical.
"""

from __future__ import annotations

import torch

from ..vocab import LETTER_IDS, NULL_ID, VOCAB_SIZE

LETTER_IDS_T = torch.tensor(LETTER_IDS, dtype=torch.long)


def letters_to_vocab(soft_letters: torch.Tensor) -> torch.Tensor:
    """(..., A) letter distributions -> (..., VOCAB_SIZE) vocab distributions
    (letter mass scattered onto letter ids; specials zero)."""
    out = soft_letters.new_zeros(*soft_letters.shape[:-1], VOCAB_SIZE)
    out[..., LETTER_IDS_T.to(soft_letters.device)] = soft_letters
    return out


def build_frame(
    slot1_letters: torch.Tensor,
    slot2_letters: torch.Tensor,
    w_uni: torch.Tensor,
) -> torch.Tensor:
    """Assemble the 2N-slot vocab-distribution frame.

    slot1_letters, slot2_letters: (N, A) row-stochastic letter distributions.
    w_uni: (N,) in [0, 1] — probability the token is a unigram (slot 2 = NULL).
    Returns (2N, VOCAB_SIZE) row-stochastic.
    """
    n = slot1_letters.shape[0]
    v1 = letters_to_vocab(slot1_letters)
    v2 = letters_to_vocab(slot2_letters)
    # Probability-space NULL blend — see module docstring for why not
    # logaddexp: p2 = w * e_NULL + (1 - w) * p_letters.
    v2 = v2 * (1.0 - w_uni)[:, None]
    v2[:, NULL_ID] = v2[:, NULL_ID] + w_uni
    frame = torch.stack([v1, v2], dim=1).reshape(2 * n, VOCAB_SIZE)
    return frame


def log_frame(frame: torch.Tensor, min_prob: float = 1e-30) -> torch.Tensor:
    """Safe log of a frame for consumers that want log-space: zeros clamp to
    log(min_prob), never producing -inf pairs downstream."""
    return torch.log(frame.clamp_min(min_prob))


def expected_embeddings(
    frame: torch.Tensor, embedding_weight: torch.Tensor
) -> torch.Tensor:
    """(2N, V) frame x (V, d) embedding table -> (2N, d) expected embeddings
    (the mixture-input path, design §8 / R3)."""
    return frame @ embedding_weight


def straight_through_frame(frame: torch.Tensor) -> torch.Tensor:
    """Straight-through fallback (design §8 hedge): forward pass sees the
    argmax one-hot, backward pass sees the soft frame's gradient."""
    hard = torch.zeros_like(frame)
    hard.scatter_(-1, frame.argmax(dim=-1, keepdim=True), 1.0)
    # (frame - frame.detach()) FIRST: exact zeros, so the forward value is
    # exactly one-hot; grouping as (hard + frame) - frame.detach() is not.
    return hard + (frame - frame.detach())
