"""Rung 2 — unigram homophonic head — task CH.5 (= 5.3 / 6.6).

Many cipher symbols → one plaintext letter, so NO Sinkhorn/permutation prior
(inverse-note §6): the parameterization is a row-stochastic soft-assignment
matrix (V_sym, A). Regularizers: row-entropy annealing (push rows one-hot)
and a letter-frequency-prior KL between the expected emitted-letter
distribution and the language's unigram prior.

Search loop (inverse-note §7/§9 hybrid): gradient steps on the soft matrix,
temperature/entropy annealing, many restarts, interleaved with a discrete
hill-climb / fixed-temperature SA over single-symbol reassignments rescored
by the exact penta-gram (`score_hard`) — the classical workhorse the
literature anchors were solved with (AZdecrypt/Kopal SA at pentagram order).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .ngram import A

# Fork-inherited state for parallel restarts (Linux fork COW: children see the
# parent's head/cipher without pickling the ~50MB LM tables per task).
_PAR: dict = {}


def _restart_job(args) -> tuple[np.ndarray, float, int]:
    seed, sa_steps, t_start, t_end = args
    torch.set_num_threads(1)
    head: HomophonicHead = _PAR["head"]
    rng = np.random.default_rng(seed)
    init = rng.integers(0, A, size=_PAR["n_symbols"])
    return head._sa_phase(
        _PAR["cipher_ids"],
        init,
        _PAR["language"],
        rng,
        steps=sa_steps,
        t_start=t_start,
        t_end=t_end,
    )


def _nice_worker() -> None:
    torch.set_num_threads(1)
    os.nice(10)  # stay polite to the Phase-A trainers


@dataclass
class Rung2Result:
    sym_to_letter: np.ndarray  # (V_sym,) argmax map
    hard_score: float  # penalized discrete objective of the winning map
    n_evals: int
    restarts_used: int
    raw_ll: float = float("nan")  # unpenalized exact n-gram log-likelihood
    # distinct restart optima, best penalized objective first:
    # [(sym_map, penalized_score, raw_ll)] — the outer tier's shortlist
    shortlist: list = field(default_factory=list)


class HomophonicHead:
    def __init__(
        self,
        evaluator,
        *,
        steps: int = 250,
        lr: float = 0.3,
        soft_order: int = 3,
        rescore_order: int = 5,
        entropy_weight: float = 0.5,
        freq_kl_weight: float = 1.0,
        freq_penalty_weight: float = 1.0,
        seed: int = 0,
    ):
        self.ev = evaluator
        self.steps = steps
        self.lr = lr
        self.soft_order = soft_order
        self.rescore_order = rescore_order
        self.entropy_weight = entropy_weight
        self.freq_kl_weight = freq_kl_weight
        # weight of the discrete objective's letter-frequency KL penalty
        # (nats per char of KL); see _objective for why it is load-bearing
        self.freq_penalty_weight = freq_penalty_weight
        self.seed = seed

    # -- gradient phase ------------------------------------------------------

    def _gradient_phase(
        self,
        cipher: torch.Tensor,
        n_symbols: int,
        language: str,
        g: torch.Generator,
    ) -> tuple[np.ndarray, int]:
        L = len(cipher)
        occ = torch.bincount(cipher, minlength=n_symbols).float()
        prior = torch.from_numpy(
            np.exp(self.ev.logT(language, 1).cpu().numpy().ravel())
        ).float()
        logits = 0.1 * torch.randn(n_symbols, A, generator=g)
        logits.requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=self.lr)
        for step in range(self.steps):
            frac = step / max(self.steps - 1, 1)
            p = torch.softmax(logits, dim=1)  # (V, A) row-stochastic
            soft_letters = p[cipher]
            ll = self.ev.score_fixed(
                soft_letters, language=language, order=self.soft_order
            )
            # expected emitted-letter distribution vs unigram prior
            emitted = (p * occ[:, None]).sum(0) / occ.sum()
            kl = (
                emitted * (torch.log(emitted.clamp_min(1e-9)) - torch.log(prior))
            ).sum()
            # row entropy, annealed up: late in training rows must commit
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(1).mean()
            loss = -ll / L + self.freq_kl_weight * kl + self.entropy_weight * frac * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
        return logits.detach().argmax(dim=1).numpy(), self.steps

    # -- discrete phase ------------------------------------------------------

    def _objective(self, decoded: np.ndarray, language: str) -> float:
        """Penalized discrete objective: exact n-gram log-likelihood MINUS a
        letter-frequency-prior KL penalty.

        The penalty is load-bearing, not cosmetic: the pure LM objective has
        DEGENERATE optima — maps sending most symbols to a few letters yield
        hyper-likely repetitive decodes that outscore the true map by >150
        nats on 408-char problems (measured). The plan's rung-2 spec calls
        for exactly this frequency prior; the classical solvers rely on the
        same constraint.
        """
        ll = self.ev.score_hard(decoded, language=language, order=self.rescore_order)
        emp = np.bincount(decoded, minlength=A) / len(decoded)
        log_prior = self.ev.logT(language, 1).cpu().numpy().ravel()
        nz = emp > 0
        kl = float((emp[nz] * (np.log(emp[nz]) - log_prior[nz])).sum())
        return ll - self.freq_penalty_weight * len(decoded) * kl

    def _sa_phase(
        self,
        cipher_np: np.ndarray,
        sym_map: np.ndarray,
        language: str,
        rng: np.random.Generator,
        steps: int = 100_000,
        t_start: float = 8.0,
        t_end: float = 0.3,
    ) -> tuple[np.ndarray, float, int]:
        """Simulated annealing over single-symbol reassignments (plus
        occasional symbol-pair letter swaps) on the penalized objective —
        the classical homophonic workhorse (AZdecrypt/Kopal lineage).
        Returns the best map ever visited, not the final one."""
        n_symbols = len(sym_map)
        score = self._objective(sym_map[cipher_np], language)
        best_map, best = sym_map.copy(), score
        n_evals = 1
        for step in range(steps):
            t = t_start * (t_end / t_start) ** (step / max(steps - 1, 1))
            if rng.random() < 0.9:  # reassign one symbol
                s = rng.integers(n_symbols)
                old = sym_map[s]
                new = rng.integers(A)
                if new == old:
                    continue
                sym_map[s] = new
                undo = [(s, old)]
            else:  # swap the letters of two symbols
                s1, s2 = rng.integers(n_symbols, size=2)
                if sym_map[s1] == sym_map[s2]:
                    continue
                undo = [(s1, sym_map[s1]), (s2, sym_map[s2])]
                sym_map[s1], sym_map[s2] = sym_map[s2], sym_map[s1]
            cand = self._objective(sym_map[cipher_np], language)
            n_evals += 1
            if cand > score or rng.random() < np.exp((cand - score) / t):
                score = cand
                if score > best:
                    best_map, best = sym_map.copy(), score
            else:
                for s, old in undo:
                    sym_map[s] = old
        # greedy polish from the best visited map
        sym_map, score = best_map, best
        improved = True
        while improved:
            improved = False
            for s in rng.permutation(n_symbols):
                cur = sym_map[s]
                for letter in range(A):
                    if letter == cur:
                        continue
                    sym_map[s] = letter
                    sc = self._objective(sym_map[cipher_np], language)
                    n_evals += 1
                    if sc > score:
                        score, cur, improved = sc, letter, True
                    else:
                        sym_map[s] = cur
        return sym_map, score, n_evals

    def polish_pairs(
        self, cipher_np: np.ndarray, sym_map: np.ndarray, language: str
    ) -> tuple[np.ndarray, float, int]:
        """Greedy polish with an enriched move set: single-symbol
        reassignments AND symbol-pair letter swaps (the CH.5 "pair-swap"
        lever for in-basin, polish-limited solves), under the penalized
        objective, until no move improves. Exhaustive: 54·25 + C(54,2)
        candidates per sweep, ~0.1 ms each."""
        sym_map = sym_map.copy()
        n_symbols = len(sym_map)
        score = self._objective(sym_map[cipher_np], language)
        n_evals = 1
        improved = True
        while improved:
            improved = False
            for s in range(n_symbols):
                cur = sym_map[s]
                for letter in range(A):
                    if letter == cur:
                        continue
                    sym_map[s] = letter
                    sc = self._objective(sym_map[cipher_np], language)
                    n_evals += 1
                    if sc > score + 1e-9:
                        score, cur, improved = sc, letter, True
                    else:
                        sym_map[s] = cur
            for s1 in range(n_symbols):
                for s2 in range(s1 + 1, n_symbols):
                    if sym_map[s1] == sym_map[s2]:
                        continue
                    sym_map[s1], sym_map[s2] = sym_map[s2], sym_map[s1]
                    sc = self._objective(sym_map[cipher_np], language)
                    n_evals += 1
                    if sc > score + 1e-9:
                        score, improved = sc, True
                    else:
                        sym_map[s1], sym_map[s2] = sym_map[s2], sym_map[s1]
        return sym_map, score, n_evals

    def _frequency_init(
        self,
        cipher_np: np.ndarray,
        n_symbols: int,
        language: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Assign symbols to letters so cumulative symbol frequency tracks the
        LM unigram distribution (homophone-flattening inverted)."""
        occ = np.bincount(cipher_np, minlength=n_symbols)
        prior = np.exp(self.ev.logT(language, 1).cpu().numpy().ravel())
        sym_map = np.zeros(n_symbols, dtype=np.int64)
        # greedy proportional fill: heaviest symbol goes to the letter whose
        # frequency quota is least satisfied, so cumulative symbol mass per
        # letter tracks the unigram prior (homophone flattening, inverted).
        quota = prior / prior.sum() * occ.sum()
        filled = np.zeros(A)
        for s in np.argsort(-occ):
            l = int(np.argmax(quota - filled))
            sym_map[s] = l
            filled[l] += occ[s]
        return sym_map

    def solve(
        self,
        cipher_ids: np.ndarray,
        n_symbols: int,
        *,
        language: str,
        restarts: int = 16,
        gradient_restarts: int = 2,
        sa_steps: int = 100_000,
        t_start: float = 15.0,
        t_end: float = 0.5,
    ) -> Rung2Result:
        """Best-of-``restarts`` SA runs. Empirically (Zodiac-408-class,
        Italian held-out) a random-init 100k-step SA run lands in the true
        basin ~1/8 of the time — the restart budget, not the objective, is
        the binding constraint (the doc's 'expect to need the restarts').
        Restart inits: random maps, plus the frequency init and (optionally)
        gradient-phase argmax maps as seeded candidates."""
        cipher_t = torch.from_numpy(cipher_ids.astype(np.int64))
        rng = np.random.default_rng(self.seed)
        best: Rung2Result | None = None
        total_evals = 0

        def consider(sym_map, n0, r):
            nonlocal best, total_evals
            sym_map, score, n1 = self._sa_phase(
                cipher_ids,
                sym_map.copy(),
                language,
                rng,
                steps=sa_steps,
                t_start=t_start,
                t_end=t_end,
            )
            total_evals += n0 + n1
            if best is None or score > best.hard_score:
                best = Rung2Result(sym_map, score, total_evals, r + 1)

        consider(self._frequency_init(cipher_ids, n_symbols, language, rng), 0, 0)
        for r in range(restarts - 1):
            if r < gradient_restarts:
                g = torch.Generator().manual_seed(self.seed + 1000 * r)
                init, n0 = self._gradient_phase(cipher_t, n_symbols, language, g)
            else:
                init, n0 = rng.integers(0, A, size=n_symbols), 0
            consider(init, n0, r + 1)
        assert best is not None
        best.n_evals = total_evals
        best.raw_ll = self.ev.score_hard(
            best.sym_to_letter[cipher_ids],
            language=language,
            order=self.rescore_order,
        )
        return best

    def solve_parallel(
        self,
        cipher_ids: np.ndarray,
        n_symbols: int,
        *,
        language: str,
        restarts: int = 48,
        workers: int = 6,
        sa_steps: int = 100_000,
        t_start: float = 15.0,
        t_end: float = 0.5,
        shortlist: int = 12,
    ) -> Rung2Result:
        """Random-restart SA fanned out over forked worker processes.

        Restart budget is the binding constraint on Zodiac-408-class
        problems (~1/12–1/36 basin-hit rate per 100k-step run measured on
        latin), so wall-clock scales down ~linearly with workers. Workers
        are single-threaded and nice(10) — the Phase-A trainers keep
        priority. Deterministic given (seed, restarts): worker seeds are
        self.seed + restart index.
        """
        _PAR.update(
            head=self,
            cipher_ids=cipher_ids,
            n_symbols=n_symbols,
            language=language,
        )
        jobs = [
            (self.seed * 100_000 + r, sa_steps, t_start, t_end) for r in range(restarts)
        ]
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_nice_worker) as pool:
            results = pool.map(_restart_job, jobs)
        best_map, best_score, n_evals = None, -np.inf, 0
        for m, s, n in results:
            n_evals += n
            if s > best_score:
                best_map, best_score = m, s
        res = Rung2Result(best_map, best_score, n_evals, restarts)
        res.raw_ll = self.ev.score_hard(
            best_map[cipher_ids], language=language, order=self.rescore_order
        )
        seen = set()
        for m, s, _ in sorted(results, key=lambda r: -r[1]):
            k = m.tobytes()
            if k in seen:
                continue
            seen.add(k)
            res.shortlist.append(
                (
                    m,
                    float(s),
                    float(
                        self.ev.score_hard(
                            m[cipher_ids], language=language, order=self.rescore_order
                        )
                    ),
                )
            )
            if len(res.shortlist) >= shortlist:
                break
        return res
