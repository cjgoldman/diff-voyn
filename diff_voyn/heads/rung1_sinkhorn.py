"""Rung 1 — 1:1 substitution head (Sinkhorn / bijective) — task CH.3 (= 5.2).

ALICE-style parameterization: a learnable square matrix, Sinkhorn-normalized
to doubly-stochastic, structurally encoding the bijective key. The search
loop is the hybrid the prototyping doc prescribes for every rung: gradient
steps on the soft map against the frozen evaluator, temperature annealing,
random restarts, then a discrete 2-swap hill-climb on the argmax permutation
rescored with the exact high-order n-gram (``score_hard``).

The head never names its evaluator — any :class:`Evaluator` implementation
drives it (the whole point of CH.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .ngram import A


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Log-space Sinkhorn normalization to a doubly-stochastic matrix."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return torch.exp(log_alpha)


@dataclass
class Rung1Result:
    sym_to_letter: np.ndarray  # (A,) argmax permutation, cipher sym -> letter
    soft_score: float
    hard_score: float  # exact n-gram-order score of the decode (nats)
    n_evals: int  # evaluator calls (R6 cost realism)
    restarts_used: int
    # distinct local optima visited, best first: [(perm, hard_score, source)]
    # — the shortlist the diffusion tier re-ranks (design §7.4 two-tier)
    shortlist: list = field(default_factory=list)


class SinkhornSubstitutionHead:
    def __init__(
        self,
        evaluator,
        *,
        gumbel_scale: float = 0.5,
        tau_start: float = 1.0,
        tau_end: float = 0.15,
        steps: int = 300,
        lr: float = 0.5,
        soft_order: int = 3,
        rescore_order: int = 5,
        seed: int = 0,
    ):
        self.ev = evaluator
        self.gumbel_scale = gumbel_scale
        self.tau_start, self.tau_end = tau_start, tau_end
        self.steps = steps
        self.lr = lr
        self.soft_order = soft_order
        self.rescore_order = rescore_order
        self.seed = seed

    def _gradient_phase(
        self, cipher: torch.Tensor, language: str, g: torch.Generator
    ) -> tuple[np.ndarray, float, int]:
        L = len(cipher)
        logits = torch.zeros(A, A, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=self.lr)
        n_evals = 0
        for step in range(self.steps):
            frac = step / max(self.steps - 1, 1)
            tau = self.tau_start * (self.tau_end / self.tau_start) ** frac
            noise = -torch.log(
                -torch.log(torch.rand(A, A, generator=g).clamp_min(1e-20))
            )
            p = sinkhorn((logits + self.gumbel_scale * (1 - frac) * noise) / tau)
            soft_letters = p[cipher]  # (L, A): row = dist over plaintext letters
            loss = (
                -self.ev.score_fixed(
                    soft_letters, language=language, order=self.soft_order
                )
                / L
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_evals += 1
        with torch.no_grad():
            cost = -sinkhorn(logits / self.tau_end).log().clamp_min(-30)
        rows, cols = linear_sum_assignment(cost.numpy())
        perm = np.empty(A, dtype=np.int64)
        perm[rows] = cols
        return perm, float(loss.detach()), n_evals

    def _swap_hillclimb(
        self, cipher_np: np.ndarray, perm: np.ndarray, language: str
    ) -> tuple[np.ndarray, float, int]:
        """Exhaustive 2-swap hill-climb under the exact high-order score."""
        best = self.ev.score_hard(
            perm[cipher_np], language=language, order=self.rescore_order
        )
        n_evals = 1
        improved = True
        while improved:
            improved = False
            for i in range(A):
                for j in range(i + 1, A):
                    cand = perm.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    s = self.ev.score_hard(
                        cand[cipher_np], language=language, order=self.rescore_order
                    )
                    n_evals += 1
                    if s > best:
                        best, perm, improved = s, cand, True
        return perm, best, n_evals

    def _ils(
        self,
        cipher_np: np.ndarray,
        perm: np.ndarray,
        language: str,
        rng: np.random.Generator,
        kicks: int = 20,
        visited: list | None = None,
    ) -> tuple[np.ndarray, float, int]:
        """Iterated local search: climb, perturb (3 random transpositions),
        re-climb, keep improvements — the classical escape from the 2-swap
        local optima that dominate short ciphers."""
        perm, best, n = self._swap_hillclimb(cipher_np, perm, language)
        if visited is not None:
            visited.append((perm.copy(), best, "ils"))
        for _ in range(kicks):
            cand = perm.copy()
            for _ in range(3):
                i, j = rng.integers(0, A, size=2)
                cand[i], cand[j] = cand[j], cand[i]
            cand, s, dn = self._swap_hillclimb(cipher_np, cand, language)
            n += dn
            if visited is not None:
                visited.append((cand.copy(), s, "ils"))
            if s > best:
                perm, best = cand, s
        return perm, best, n

    def _frequency_init(self, cipher_np: np.ndarray, language: str) -> np.ndarray:
        """Rank-match cipher symbol frequencies to LM unigram frequencies."""
        occ = np.bincount(cipher_np, minlength=A)
        prior = self.ev.logT(language, 1).cpu().numpy().ravel()
        perm = np.empty(A, dtype=np.int64)
        perm[np.argsort(-occ)] = np.argsort(-prior)
        return perm

    def solve(
        self,
        cipher_ids: np.ndarray,
        *,
        language: str,
        restarts: int = 4,
        kicks: int = 20,
        target_score: float | None = None,
        shortlist: int = 8,
    ) -> Rung1Result:
        """``shortlist``: keep that many distinct local optima (best first)
        in ``Rung1Result.shortlist`` for the outer (diffusion) tier."""
        cipher_t = torch.from_numpy(cipher_ids.astype(np.int64))
        rng = np.random.default_rng(self.seed)
        best: Rung1Result | None = None
        visited: list = []

        def consider(perm, soft, n_evals, r):
            nonlocal best
            perm, hard, n2 = self._ils(cipher_ids, perm, language, rng, kicks, visited)
            if best is None or hard > best.hard_score:
                best = Rung1Result(perm, soft, hard, n_evals + n2, r + 1)

        # restart 0: pure frequency-rank init (no gradient phase, near-free)
        consider(self._frequency_init(cipher_ids, language), float("nan"), 0, 0)
        for r in range(restarts):
            if target_score is not None and best.hard_score >= target_score:
                break
            g = torch.Generator().manual_seed(self.seed + 1000 * r)
            perm, soft, n1 = self._gradient_phase(cipher_t, language, g)
            consider(perm, -soft, n1, r + 1)
        assert best is not None
        seen, short = set(), []
        for perm, hard, src in sorted(visited, key=lambda v: -v[1]):
            key = perm.tobytes()
            if key in seen:
                continue
            seen.add(key)
            short.append((perm, float(hard), src))
            if len(short) >= shortlist:
                break
        best.shortlist = short
        return best
