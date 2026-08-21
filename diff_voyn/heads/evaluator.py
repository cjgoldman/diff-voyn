"""The frozen ``Evaluator`` contract + ``NgramEvaluator`` — task CH.1.

Every cipher head is written against this narrow interface (prototyping doc
§2): given soft (row-stochastic) distributions over the plaintext alphabet,
return a differentiable scalar plaintext-quality score. Nothing in a head may
name "n-gram" or "diffusion" — the evaluator swap (post-G4) must be a
constructor argument, not a rewrite.

Calibration hook (§6): per-language additive offsets are applied in exactly
ONE place — :meth:`EvaluatorBase.calibrated_bits_per_char`. The
``NgramEvaluator`` carries its own offsets; the ``DiffusionEvaluator`` later
slots its §3.4 offsets into the identical hook.

Soft-score semantics
--------------------
``score_fixed`` returns the *chained expectation* surrogate
``sum_t E_{c ~ q_t-k+1..t}[ log p(c_t | context) ]`` (positions independent
under q). It lower-bounds the true expected log-likelihood and is the
standard differentiable relaxation (ALICE / Kambhatla lineage). Hard
(argmax) maps should be re-scored with :meth:`NgramEvaluator.score_hard`,
which is exact and cheap — the discrete-move search loops use that.

``score_segmental`` marginalizes latent segmentation *and* soft letter
identity jointly: the semi-Markov forward DP over trigram state (a, b) =
last two letters. Token emissions are ``TokenEmission`` (unigram dist and/or
prefix+suffix dists with a unigram-vs-bigram weight). The unigram/bigram
blend is a true log-space mixture — the ``logaddexp(-inf, -inf)`` trap
(design §8) is guarded by masking impossible branches *before* the blend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .ngram import A, NgramLM

NEG_INF = float("-inf")


@dataclass
class TokenEmission:
    """Per-ciphertext-token candidate structure for segmental scoring.

    ``log_w_uni``/``log_w_bi`` are log mixture weights (need not sum to 1 —
    heads may put structural-prior mass here). A branch with weight -inf or a
    ``None`` distribution is impossible and is skipped, never blended.

    ``branches``, when set, overrides the uni/pre/suf convenience fields with
    an explicit branch list ``[(log_w, [dist, ...]), ...]`` — each branch
    emits its distributions in order (1..k letters). Rung 3 needs this for
    tokens with several feasible (prefix, suffix) splits.
    """

    uni: torch.Tensor | None = None  # (A,) row-stochastic
    pre: torch.Tensor | None = None  # (A,)
    suf: torch.Tensor | None = None  # (A,)
    log_w_uni: torch.Tensor | float = 0.0
    log_w_bi: torch.Tensor | float = NEG_INF
    branches: list[tuple[torch.Tensor | float, list[torch.Tensor]]] | None = None

    def iter_branches(self):
        if self.branches is not None:
            yield from self.branches
            return
        if self.uni is not None:
            yield self.log_w_uni, [self.uni]
        if self.pre is not None and self.suf is not None:
            yield self.log_w_bi, [self.pre, self.suf]


@runtime_checkable
class Evaluator(Protocol):
    """A frozen plaintext-quality scorer (design §7.4). No parameters are
    ever updated through this object."""

    A: int
    languages: list[str]

    def score_fixed(self, soft_letters: torch.Tensor, *, language: str) -> torch.Tensor:
        """(L, A) row-stochastic -> differentiable scalar log P (nats)."""
        ...

    def score_segmental(
        self, emissions: list[TokenEmission], *, language: str
    ) -> torch.Tensor:
        """Token emissions -> scalar log-likelihood, marginalized over
        segmentation via the semi-Markov forward DP."""
        ...

    def as_embedding_frame(
        self, soft_letters: torch.Tensor, null_weights: torch.Tensor
    ) -> torch.Tensor:
        """Project onto the 2N-slot expected-embedding frame (design §8).
        Only the DiffusionEvaluator consumes this; others may raise."""
        ...


class EvaluatorBase:
    """Shared calibration hook — the single place offsets are applied."""

    A: int = A
    languages: list[str]
    # bits/char, ADDED to the raw bits/char at ranking time (§3.4 / 5b).
    calibration_offsets_bits: dict[str, float]

    def calibrated_bits_per_char(
        self, raw_score_nats: torch.Tensor | float, n_chars: float, language: str
    ) -> float:
        """Raw log-likelihood-scale score → calibrated bits/char. The
        arithmetic lives in :func:`diff_voyn.metrology.calibration.
        calibrate_bits` (task 3.4 "applied in exactly one place")."""
        from ..metrology.calibration import calibrate_bits

        raw = float(raw_score_nats)
        bits = -raw / (n_chars * float(np.log(2.0)))
        return calibrate_bits(bits, language, self.calibration_offsets_bits)


class NgramEvaluator(EvaluatorBase):
    """The workhorse / permanent inner-loop scorer (prototyping doc §2).

    ``dp_order`` bounds the segmental DP state space (trigram -> 625 states);
    ``fixed_order`` is the default for fixed-alignment scoring (pentagram for
    the literature anchors).
    """

    def __init__(
        self,
        lms: dict[str, NgramLM],
        *,
        dp_order: int = 3,
        fixed_order: int = 5,
        device: str | torch.device = "cpu",
        calibration_offsets_bits: dict[str, float] | None = None,
    ):
        self.lms = lms
        self.languages = sorted(lms)
        self.dp_order = dp_order
        self.fixed_order = fixed_order
        self.device = torch.device(device)
        self.calibration_offsets_bits = dict(calibration_offsets_bits or {})
        self._t_cache: dict[tuple[str, int], torch.Tensor] = {}

    # -- table access -------------------------------------------------------

    def logT(self, language: str, order: int) -> torch.Tensor:
        """(A,)*order log-prob tensor on ``self.device`` (frozen, no grad)."""
        key = (language, order)
        if key not in self._t_cache:
            t = torch.from_numpy(self.lms[language].table(order)).to(self.device)
            self._t_cache[key] = t.reshape((A,) * order)
        return self._t_cache[key]

    # -- fixed alignment ----------------------------------------------------

    def score_fixed(
        self,
        soft_letters: torch.Tensor,
        *,
        language: str,
        order: int | None = None,
        chunk: int = 64,
    ) -> torch.Tensor:
        """Chained-expectation log-likelihood of an (L, A) soft stream.

        Warm-up positions (t < order-1) are scored with the highest order
        their history allows, so every position contributes exactly once.
        """
        k = order or self.fixed_order
        L = soft_letters.shape[0]
        q = soft_letters.to(self.device)
        total = q.new_zeros(())
        for kk in range(1, min(k, L + 1)):  # position kk-1 scored at order kk
            total = total + self._soft_window(q[:kk].unsqueeze(0), language, kk)[0]
        if L < k:
            return total
        # windows [t-k+1 .. t] for t = k-1 .. L-1
        w = q.unfold(0, k, 1)  # (W, A, k) — unfold appends the window dim
        w = w.permute(0, 2, 1)  # (W, k, A)
        for i in range(0, w.shape[0], chunk):
            total = total + self._soft_window(w[i : i + chunk], language, k).sum()
        return total

    def _soft_window(self, w: torch.Tensor, language: str, k: int) -> torch.Tensor:
        """w: (W, k, A) -> (W,) expected log-prob of the last letter given the
        first k-1, contracting the order-k table one context dim at a time."""
        W = w.shape[0]
        # Progressive contraction, oldest context dim first. The first step is
        # a plain matmul against the shared table (no per-window expansion).
        x = w[:, 0, :] @ self.logT(language, k).reshape(A, -1)  # (W, A^(k-1))
        for j in range(1, k):
            rest = A ** (k - 1 - j)
            # contract leading letter dim: (W, A, rest) x (W, A) -> (W, rest)
            x = torch.einsum("war,wa->wr", x.reshape(W, A, rest), w[:, j, :])
        return x.squeeze(-1)

    def score_hard(
        self, letter_ids: np.ndarray, *, language: str, order: int | None = None
    ) -> float:
        """Exact log-prob (nats) of a hard letter-index stream — the cheap
        rescoring path for discrete moves / argmax maps."""
        return self.lms[language].score_ids(letter_ids, order or self.fixed_order)

    # -- segmental (semi-Markov forward DP) ---------------------------------

    def _advance(
        self, alpha: torch.Tensor, q: torch.Tensor, logT3: torch.Tensor
    ) -> torch.Tensor:
        """One-letter DP advance. alpha: (A, A) log-mass over (a, b) = last
        two letters. Returns alpha': (A, A) over (b, c), emitting soft q."""
        # inner[b, c] = logsumexp_a alpha[a, b] + logT3[a, b, c]
        inner = torch.logsumexp(alpha[:, :, None] + logT3, dim=0)
        return inner + torch.log(q.clamp_min(1e-45))[None, :]

    def score_segmental(
        self, emissions: list[TokenEmission], *, language: str
    ) -> torch.Tensor:
        """Forward DP over trigram state (a, b) = last two emitted letters.

        Start state: stationary bigram log p1(a) + log p2(b|a) — ciphertext
        excerpts are mid-stream (whitespace-stripped continuous text), so the
        first real letters are scored against marginalized phantom context
        rather than a sentence-start model. Every emitted letter then pays
        its trigram transition; branch mixtures blend only feasible branches
        (the ``logaddexp(-inf,-inf)`` guard, design §8).
        """
        logT3 = self.logT(language, min(self.dp_order, 3))
        if logT3.dim() != 3:
            raise NotImplementedError("segmental DP is trigram-state")
        dev = logT3.device
        alpha = self.logT(language, 1)[:, None] + self.logT(language, 2)
        for em in emissions:
            outs = []
            for log_w, dists in em.iter_branches():
                if not _finite_w(log_w):
                    continue
                b = alpha
                for q in dists:
                    b = self._advance(b, q.to(dev), logT3)
                outs.append(b + _as_t(log_w, dev))
            if not outs:
                raise ValueError("token with no feasible branch (all -inf)")
            alpha = outs[0]
            for b in outs[1:]:
                alpha = torch.logaddexp(alpha, b)
        return torch.logsumexp(alpha.reshape(-1), dim=0)

    # -- char-lattice segmental DP (rung 4) ----------------------------------
    #
    # Like ``score_hard``, these are NgramEvaluator-only inner-loop scorers
    # (not part of the frozen protocol): the rung-4 head's segmentation is
    # latent over CHAR positions, not pre-parsed tokens, so the DP runs over a
    # (position, segment-length) lattice. Post-G4, rung 4 keeps this n-gram DP
    # as its inner search and rescores shortlisted hard decodes with the
    # diffusion evaluator (the design's shortlist convention).

    def score_lattice(
        self,
        log_emis: torch.Tensor,
        seg_lengths: list[int],
        *,
        language: str,
        order: int = 3,
        start_window: int = 1,
        end_window: int = 1,
    ) -> torch.Tensor:
        """Marginal log-likelihood over segmentations of a char stream.

        ``log_emis[i, j, :]`` is the (A,) log emission weight of the segment
        starting at char ``i`` with length ``seg_lengths[j]`` (use a large
        negative constant, NOT -inf, for inadmissible segments — an all
        -inf ``logsumexp`` NaNs in backward). Differentiable w.r.t.
        ``log_emis``. ``order`` is 2 (state = last letter) or 3.

        ``start_window``/``end_window`` > 1 let the parse begin within the
        first / terminate within the last that many positions instead of
        covering all L chars exactly — the right semantics for a mid-stream
        chunk whose edges need not align with segment boundaries (an
        edge-misaligned dead lattice otherwise returns the ~-1e30 sentinel
        and its gradient blows up the caller's parameters).
        """
        L = int(log_emis.shape[0])
        neg = torch.full((), -1e30, dtype=log_emis.dtype)
        if order == 2:
            logT2 = self.logT(language, 2).to(log_emis.dtype)
            start2 = self.logT(language, 1).to(log_emis.dtype)
            alphas: list[torch.Tensor] = [start2]
            for i in range(1, L + 1):
                outs = [
                    torch.logsumexp(alphas[i - n][:, None] + logT2, dim=0)
                    + log_emis[i - n, j]
                    for j, n in enumerate(seg_lengths)
                    if i - n >= 0
                ]
                if i < start_window:  # parse may begin at position i
                    outs.append(start2)
                alphas.append(_stack_lse(outs, neg, (A,), log_emis.dtype))
        elif order == 3:
            logT3 = self.logT(language, 3).to(log_emis.dtype)
            start = (self.logT(language, 1)[:, None] + self.logT(language, 2)).to(
                log_emis.dtype
            )
            alphas = [start]
            for i in range(1, L + 1):
                outs = []
                for j, n in enumerate(seg_lengths):
                    if i - n < 0:
                        continue
                    inner = torch.logsumexp(alphas[i - n][:, :, None] + logT3, dim=0)
                    outs.append(inner + log_emis[i - n, j][None, :])
                if i < start_window:
                    outs.append(start)
                alphas.append(_stack_lse(outs, neg, (A, A), log_emis.dtype))
        else:
            raise NotImplementedError("lattice DP supports order 2 or 3")
        ends = [alphas[i].reshape(-1) for i in range(max(L - end_window + 1, 1), L + 1)]
        return torch.logsumexp(torch.cat(ends), dim=0)

    def viterbi_lattice(
        self,
        log_emis: torch.Tensor,
        seg_lengths: list[int],
        *,
        language: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Max-probability segmentation + letters (trigram state).

        Returns (segment start positions, letter ids, path log-prob).
        """
        with torch.no_grad():
            logT3 = self.logT(language, 3).to(log_emis.dtype)
            L = int(log_emis.shape[0])
            beta = [
                (self.logT(language, 1)[:, None] + self.logT(language, 2)).to(
                    log_emis.dtype
                )
            ]
            back: list[tuple | None] = [None]
            for i in range(1, L + 1):
                cands, metas = [], []
                for j, n in enumerate(seg_lengths):
                    if i - n < 0:
                        continue
                    # (a, b, c) -> max over a
                    x = beta[i - n][:, :, None] + logT3
                    inner, arg_a = x.max(dim=0)
                    cands.append(inner + log_emis[i - n, j][None, :])
                    metas.append((n, arg_a))
                if not cands:
                    beta.append(torch.full((A, A), -1e30, dtype=log_emis.dtype))
                    back.append(None)
                    continue
                stacked = torch.stack(cands)  # (k, A, A)
                best, arg_k = stacked.max(dim=0)
                beta.append(best)
                back.append((metas, arg_k))
            flat = int(beta[L].argmax())
            b, c = divmod(flat, A)
            score = float(beta[L][b, c])
            i, starts, letters = L, [], []
            while i > 0 and back[i] is not None:
                metas, arg_k = back[i]
                k = int(arg_k[b, c])
                n, arg_a = metas[k]
                starts.append(i - n)
                letters.append(c)
                a = int(arg_a[b, c])
                i, b, c = i - n, a, b
            return (
                np.array(starts[::-1], dtype=np.int64),
                np.array(letters[::-1], dtype=np.int64),
                score,
            )

    # -- diffusion-only path -------------------------------------------------

    def as_embedding_frame(self, soft_letters, null_weights):  # pragma: no cover
        raise NotImplementedError("n-gram evaluator does not consume the frame")


def _stack_lse(
    outs: list[torch.Tensor], neg: torch.Tensor, shape: tuple, dtype
) -> torch.Tensor:
    if not outs:
        return neg.expand(shape).to(dtype).clone()
    if len(outs) == 1:
        return outs[0]
    return torch.logsumexp(torch.stack(outs), dim=0)


def _finite_w(w) -> bool:
    return (
        float(w) > NEG_INF if not torch.is_tensor(w) else bool(torch.isfinite(w).item())
    )


def _as_t(w, device) -> torch.Tensor:
    return w if torch.is_tensor(w) else torch.tensor(float(w), device=device)
