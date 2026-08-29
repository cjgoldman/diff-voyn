"""Rung 4 — arithmetic sum-to-target head (task CH.8 / 5.5, design §10).

Target cipher: ``voynpy.pseudo_vms`` (pinned) — each plaintext letter carries
an integer value; a cipher token is 2–6 chars from a 16-char alphabet whose
(unknown) integer char values sum to the letter's value. Our pipeline strips
whitespace, so the head sees an UNSEGMENTED char stream and must marginalize
the token segmentation (the doc's "generalize the Naibbe semi-Markov DP").

Key (learned): ``v`` — 16 char values; ``u`` — per-letter values. A segment
decodes to the letter whose value equals the segment's sum. Structural prior
(published apparatus, not key material): the 2–6 length range with the
VMS-calibrated length distribution, and the canonical WITHIN-TOKEN char
ordering — every token is sorted by a fixed global order on the 16 chars
(negatives first high-to-low, then positives high-to-low). Two consequences
the head exploits:

1. The global order is identifiable from ciphertext alone: an adjacent pair
   (x, y) with y before x in the order can only straddle a token boundary, so
   pairwise precedence counts recover the order (``infer_char_order``); any
   descent then FORCES a boundary, pruning the segment lattice hard.
2. Under the scheme's value convention (negatives −1..−s prefix, positives
   descending M..0, 16 consecutive integers) the inferred order determines
   ``v`` up to the single split parameter s: v = (−1..−s, 15−s, 14−s, .., 0).
   These order-derived candidates are the restart initializations; ``v``
   stays free so gradient descent can leave the convention if the data says
   so (the plan's "jointly infer" requirement).

Scoring: emissions on the (position, length) lattice index a Gumbel–Sinkhorn
assignment between integer segment sums and letters (the gradient phase) or
a small-σ Gaussian kernel around integer keys (the discrete phase);
``NgramEvaluator.score_lattice`` (bigram state for the inner loop, trigram
for final scoring) marginalizes segmentation × letters.

On the rung-2 frequency-KL defense: measured here, the cheap
emission-posterior proxy for decoded letter frequency MIS-TARGETS the true
key (~320 nats of penalty on a 300-letter instance — the average over all
admissible, mostly spurious, segments diverges from the LM prior even under
the truth), and with the penalty on, found keys outscore the truth. With it
off the truth outscores every found key and u-accuracy improves; the
injectivity of the Sinkhorn/Hungarian assignment is itself the
degenerate-map defense rung 2 lacked. Default ``freq_penalty_weight=0``;
the knob remains for experiments with an exact decoded-frequency penalty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from numba import njit

from .ngram import A

N_SYM = 16
SEG_LENGTHS = [2, 3, 4, 5, 6]
# VMS-calibrated token-length distribution (upstream pinned config; public).
LENGTH_PRIOR = {2: 0.10, 3: 0.22, 4: 0.26, 5: 0.26, 6: 0.16}
NEG = -1e30


# -- structural inference ----------------------------------------------------


def _lop_order(w: np.ndarray) -> np.ndarray:
    """Exact linear ordering: permutation maximizing sum of w[x, y] over
    pairs with x placed before y (subset DP, 2^n states — n=16 is cheap)."""
    n = w.shape[0]
    w = w.astype(np.float64).copy()
    np.fill_diagonal(w, 0.0)
    rowtot = w.sum(axis=1)
    full = 1 << n
    # subsum[z][S] = sum of w[z, y] for y in S
    subsum = np.zeros((n, full), dtype=np.float64)
    for s in range(1, full):
        low = s & -s
        j = low.bit_length() - 1
        rest = s ^ low
        subsum[:, s] = subsum[:, rest] + w[:, j]
    best = np.full(full, -np.inf)
    best[0] = 0.0
    choice = np.zeros(full, dtype=np.int64)
    for s in range(1, full):
        b, arg = -np.inf, 0
        for z in range(n):
            bit = 1 << z
            if not s & bit:
                continue
            prev = s ^ bit
            # placing z after the chars in `prev`: it precedes every y not
            # yet placed, contributing rowtot[z] - subsum[z, prev]
            cand = best[prev] + rowtot[z] - subsum[z, prev]
            if cand > b:
                b, arg = cand, z
        best[s], choice[s] = b, arg
    order = []
    s = full - 1
    while s:
        z = int(choice[s])
        order.append(z)
        s ^= 1 << z
    order = order[::-1]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def _boundary_score(
    char_ids: np.ndarray, rank: np.ndarray, n_tok: float, n_sym: int = N_SYM
) -> float:
    """Log-likelihood of the descending adjacencies under a factorized
    boundary model P_end(x) x P_start(y) fitted on those same descents.
    Descents are FORCED boundaries under the candidate order, so a wrong
    order must explain systematic within-token pairs as boundaries — which
    the factorization cannot do cheaply. -inf if the order forces more
    boundaries than tokens exist, or forces two ADJACENT boundaries — a
    length-1 token, impossible under the min token length of 2. (The latter
    kills cyclic rotations of the true order, which reinterpret the
    boundary wrap-around and otherwise fit the factorized model well.)"""
    r = rank[char_ids]
    desc = r[1:] < r[:-1]
    if int(desc.sum()) > n_tok - 1 or bool(np.any(desc[1:] & desc[:-1])):
        return -np.inf  # n_tok is the min-length bound: L / min token length
    ends = np.concatenate([char_ids[:-1][desc], char_ids[-1:]])
    starts = np.concatenate([char_ids[1:][desc], char_ids[:1]])
    p_end = (np.bincount(ends, minlength=n_sym) + 0.5) / (len(ends) + 0.5 * n_sym)
    p_start = (np.bincount(starts, minlength=n_sym) + 0.5) / (len(starts) + 0.5 * n_sym)
    return float(
        np.sum(np.log(p_end[char_ids[:-1][desc]]))
        + np.sum(np.log(p_start[char_ids[1:][desc]]))
    )


def infer_char_orders(
    char_ids: np.ndarray,
    *,
    n_sym: int = N_SYM,
    top_k: int = 3,
    n_samples: int = 60,
    seed: int = 0,
) -> list[np.ndarray]:
    """Candidate global canonical char orders, best first.

    Within-token adjacent pairs always respect the order (~3/4 of all
    adjacencies at mean token length ~4), so the true order is a linear-
    ordering optimum of the adjacency counts — but that optimum is heavily
    degenerate: a char appearing only token-finally (the '0' analogue) can
    be moved at zero cost, because boundary pairs are directional (final
    chars are low positives, initial chars negatives / high positives), and
    pairs that never co-occur within a token are ordered by boundary counts
    alone. Ties are broken by ``_boundary_score``: sample the optimum class
    with Gumbel-noised exact LOP solves, score each candidate, return the
    top-k distinct orders (measured: the true order is top-1 in most
    instances and within the top few otherwise — the caller carries the
    candidates as restart seeds and lets the DP objective select).
    """
    counts = np.zeros((n_sym, n_sym), dtype=np.float64)
    np.add.at(counts, (char_ids[:-1], char_ids[1:]), 1.0)
    np.fill_diagonal(counts, 0.0)
    rng = np.random.default_rng(seed)
    pool = {tuple(_lop_order(counts))}
    for _ in range(n_samples):
        noise = rng.gumbel(0, np.sqrt(counts + 1.0) * 0.5)
        pool.add(tuple(_lop_order(counts + noise)))
    n_tok_max = len(char_ids) / SEG_LENGTHS[0]
    scored = sorted(
        (
            (_boundary_score(char_ids, np.array(o), n_tok_max, n_sym=n_sym), o)
            for o in pool
        ),
        key=lambda x: -x[0],
    )
    return [np.array(o) for s, o in scored[:top_k] if np.isfinite(s)]


def infer_char_order(char_ids: np.ndarray, n_sym: int = N_SYM) -> np.ndarray:
    """Single best candidate order (see ``infer_char_orders``)."""
    return infer_char_orders(char_ids, n_sym=n_sym, top_k=1)[0]


def segmented_admissible_mask(
    char_ids: np.ndarray, token_starts: np.ndarray
) -> np.ndarray:
    """(L, len(SEG_LENGTHS)) bool for an OBSERVED segmentation (task 6.1 —
    the manuscript's word boundaries are visible, so the rung-4 lattice
    needs no latent segmentation): segment [i, i+n) is admissible iff i is
    a token start and i+n is the next start (or the end of the stream).
    Tokens whose length is outside ``SEG_LENGTHS`` leave no admissible
    segment — drop them upstream or the lattice has no path."""
    L = len(char_ids)
    starts = np.asarray(token_starts, dtype=np.int64)
    ends = np.concatenate([starts[1:], [L]])
    adm = np.zeros((L, len(SEG_LENGTHS)), dtype=bool)
    for s, e in zip(starts, ends):
        n = int(e - s)
        if n in SEG_LENGTHS:
            adm[s, SEG_LENGTHS.index(n)] = True
    return adm


def positional_rank(
    char_ids: np.ndarray, token_starts: np.ndarray, n_sym: int = N_SYM
) -> np.ndarray:
    """Char order from mean within-token relative position (initial → first)
    — the segmented analogue of ``infer_char_order``, used only to seed
    ``order_derived_values``; the gradient / polish phases move ``v``."""
    L = len(char_ids)
    starts = np.asarray(token_starts, dtype=np.int64)
    ends = np.concatenate([starts[1:], [L]])
    pos_sum = np.zeros(n_sym)
    cnt = np.zeros(n_sym)
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        rel = np.arange(n) / (n - 1)
        np.add.at(pos_sum, char_ids[s:e], rel)
        np.add.at(cnt, char_ids[s:e], 1.0)
    mean_pos = np.where(cnt > 0, pos_sum / np.maximum(cnt, 1), 0.5)
    order = np.argsort(mean_pos, kind="stable")
    rank = np.empty(n_sym, dtype=np.int64)
    rank[order] = np.arange(n_sym)
    return rank


def admissible_mask(char_ids: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """(L, len(SEG_LENGTHS)) bool: segment [i, i+n) is admissible iff it fits
    in the stream and is non-descending in the canonical order."""
    L = len(char_ids)
    r = rank[char_ids]
    ascending = np.concatenate([r[1:] >= r[:-1], [False]])  # ok[i]: (i, i+1)
    run_ok = np.ones((L, len(SEG_LENGTHS)), dtype=bool)
    for j, n in enumerate(SEG_LENGTHS):
        ok = np.zeros(L, dtype=bool)
        valid = L - n
        if valid >= 0:
            acc = np.ones(valid + 1, dtype=bool)
            for d in range(n - 1):
                acc &= ascending[d : d + valid + 1]
            ok[: valid + 1] = acc
        run_ok[:, j] = ok
    return run_ok


def order_derived_values(rank: np.ndarray, split: int) -> np.ndarray:
    """Candidate integer char values from the inferred order: the first
    ``split`` chars are negatives −1..−split, the rest positives descending
    to 0 (the scheme's consecutive-integer convention; 16 chars & split 2
    give the upstream −2..13)."""
    n = len(rank)
    v = np.empty(n, dtype=np.float64)
    for c in range(n):
        p = rank[c]
        v[c] = -(p + 1) if p < split else (n - 1 - split) - (p - split)
    return v


# -- result ------------------------------------------------------------------


@dataclass
class Rung4Result:
    v: np.ndarray  # (16,) integer char values
    u: np.ndarray  # (A,) integer letter values
    score: float  # penalized objective (comparison key)
    raw_ll: float  # trigram lattice LL of the final key
    decoded: np.ndarray  # Viterbi letter ids
    seg_starts: np.ndarray
    n_evals: int
    restarts_used: int
    wall_s: float
    extra: dict = field(default_factory=dict)
    # every restart's final key + decode, best score first — the outer
    # tier's shortlist: [(v, u, decoded, score, raw_ll, rank)]
    shortlist: list = field(default_factory=list)

    def v_accuracy(self, true_v: np.ndarray) -> float:
        return float(np.mean(self.v == true_v))

    def u_accuracy(self, true_u: np.ndarray, occ: np.ndarray | None = None) -> float:
        good = (self.u == true_u).astype(float)
        if occ is None:
            return float(good.mean())
        return float((good * occ).sum() / max(occ.sum(), 1))


# -- the head ----------------------------------------------------------------


class ArithmeticHead:
    """Sum-constrained homophonic inverse over a latent char segmentation."""

    def __init__(
        self,
        evaluator,
        *,
        steps: int = 400,
        lr: float = 0.3,
        tau_start: float = 1.0,
        tau_end: float = 0.2,
        gumbel_scale: float = 0.3,
        sigma_end: float = 0.35,
        chunk_chars: int = 768,
        freq_penalty_weight: float = 0.0,
        polish_rounds: int = 3,
        grad_order: int = 2,
        seed: int = 0,
    ):
        self.ev = evaluator
        self.steps = steps
        self.lr = lr
        self.tau_start, self.tau_end = tau_start, tau_end
        self.gumbel_scale = gumbel_scale
        self.sigma_end = sigma_end
        self.chunk_chars = chunk_chars
        self.freq_penalty_weight = freq_penalty_weight
        self.polish_rounds = polish_rounds
        self.grad_order = grad_order
        self.seed = seed
        self.len_log_prior = torch.tensor(
            [np.log(LENGTH_PRIOR[n]) for n in SEG_LENGTHS], dtype=torch.float32
        )

    # -- emissions ----------------------------------------------------------

    def _seg_sums(self, v: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        """(L, n_len) segment value sums via cumsum differences."""
        vals = v[char_ids]
        cs = torch.cat([vals.new_zeros(1), torch.cumsum(vals, 0)])
        L = len(char_ids)
        out = vals.new_full((L, len(SEG_LENGTHS)), 0.0)
        for j, n in enumerate(SEG_LENGTHS):
            if L - n >= 0:
                out[: L - n + 1, j] = cs[n:] - cs[: L - n + 1]
        return out

    def _log_emissions(
        self,
        v: torch.Tensor,
        u: torch.Tensor,
        sigma: float,
        char_ids: torch.Tensor,
        adm: torch.Tensor,
    ) -> torch.Tensor:
        """(L, n_len, A) lattice emissions; inadmissible slots get NEG."""
        V = self._seg_sums(v, char_ids)  # (L, n_len)
        d2 = (V[:, :, None] - u[None, None, :]) ** 2
        logb = self.len_log_prior[None, :, None] - d2 / (2 * sigma**2) - np.log(sigma)
        return torch.where(adm[:, :, None], logb, torch.full((), NEG))

    # -- Sinkhorn assignment phase (u given integer v) -----------------------
    #
    # With v integer (order-derived), every segment sum lies on a small
    # integer grid, so u is an ASSIGNMENT between grid values and letters —
    # the rung-1/3 setting, not a metric regression. A scalar-u Gaussian
    # kernel cannot travel the ~8 value units a frequency init is typically
    # off by (measured); a Gumbel–Sinkhorn partial permutation over
    # (grid values × letters + dummy columns for unused values), scored by
    # the lattice DP and projected with Hungarian, moves probability mass
    # instead of values through space.

    def _sum_grid(self, v: np.ndarray, char_ids: np.ndarray, adm: np.ndarray):
        """Integer grid of admissible segment sums + (L, n_len) grid-index
        array (-1 where inadmissible / off grid after rounding)."""
        vals = np.rint(v)[char_ids].astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(vals)])
        L = len(char_ids)
        sums = np.full((L, len(SEG_LENGTHS)), 10**6, dtype=np.int64)
        for j, n in enumerate(SEG_LENGTHS):
            if L - n >= 0:
                sums[: L - n + 1, j] = cs[n:] - cs[: L - n + 1]
        obs = sums[adm & (sums < 10**6)]
        lo, hi = int(obs.min()), int(obs.max())
        grid = np.arange(lo, hi + 1)
        idx = np.where(adm, sums - lo, -1)
        return grid, idx

    def _grid_mass(self, grid, idx, adm) -> np.ndarray:
        w = np.array([LENGTH_PRIOR[n] for n in SEG_LENGTHS])
        mass = np.zeros(len(grid))
        for j in range(len(SEG_LENGTHS)):
            ok = adm[:, j]
            np.add.at(mass, idx[ok, j], w[j])
        return mass

    def _gradient_phase(self, char_ids_np, adm_np, language, v0, g):
        from .rung1_sinkhorn import sinkhorn

        grid, idx_np = self._sum_grid(v0, char_ids_np, adm_np)
        n_grid = max(len(grid), A + 1)
        idx = torch.from_numpy(np.clip(idx_np, 0, None))
        adm = torch.from_numpy(adm_np)
        # affinity init: value-mass quantile vs letter-frequency quantile
        # (frequent letters get frequent sums, softly)
        mass = self._grid_mass(grid, idx_np, adm_np)
        p1 = np.exp(self.ev.logT(language, 1).numpy())
        q_val = np.argsort(np.argsort(-mass)) / max(len(grid) - 1, 1)
        q_let = np.argsort(np.argsort(-p1)) / (A - 1)
        aff = np.zeros((n_grid, n_grid), dtype=np.float32)
        aff[: len(grid), :A] = -4.0 * (q_val[:, None] - q_let[None, :]) ** 2
        logits = torch.tensor(aff) + 0.1 * torch.randn(n_grid, n_grid, generator=g)
        logits.requires_grad_(True)
        prior = torch.exp(self.ev.logT(language, 1)).float()
        opt = torch.optim.Adam([logits], lr=self.lr)
        L = len(char_ids_np)
        n_evals = 0
        for step in range(self.steps):
            frac = step / max(self.steps - 1, 1)
            tau = self.tau_start * (self.tau_end / self.tau_start) ** frac
            noise = -torch.log(
                -torch.log(torch.rand(n_grid, n_grid, generator=g).clamp_min(1e-20))
            )
            P = sinkhorn((logits + self.gumbel_scale * (1 - frac) * noise) / tau)
            log_rows = P[:, :A].clamp_min(1e-30).log()  # (n_grid, A)
            if L > self.chunk_chars:
                s0 = int(torch.randint(L - self.chunk_chars + 1, (1,), generator=g))
                sl = slice(s0, s0 + self.chunk_chars)
            else:
                sl = slice(0, L)
            idx_c, adm_c = idx[sl], adm[sl]
            logb = self.len_log_prior[None, :, None] + log_rows[idx_c]
            logb = torch.where(adm_c[:, :, None], logb, torch.full((), NEG))
            # chunk edges rarely align with segment boundaries — marginalize
            # the start / terminal position over a max-length window
            ll = self.ev.score_lattice(
                logb,
                SEG_LENGTHS,
                language=language,
                order=self.grad_order,
                start_window=SEG_LENGTHS[-1],
                end_window=SEG_LENGTHS[-1],
            )
            n_letters = len(idx_c) / 4.2  # expected letters under the length prior
            # soft decoded-frequency proxy: emission-posterior letter mass
            post = torch.softmax(logb[adm_c], dim=-1)
            p_hat = post.mean(dim=0).clamp_min(1e-9)
            kl = (p_hat * (p_hat.log() - prior.log())).sum()
            loss = -ll / n_letters + self.freq_penalty_weight * kl
            opt.zero_grad()
            # skip dead-lattice chunks (sentinel-magnitude LL) — one
            # poisoned step otherwise wrecks the logits
            if torch.isfinite(loss) and float(ll.detach()) > -1e20:
                loss.backward()
                if logits.grad is not None and torch.isfinite(logits.grad).all():
                    opt.step()
            n_evals += 1
        return grid, logits.detach(), n_evals

    def _project_u(self, grid: np.ndarray, logits: torch.Tensor) -> np.ndarray:
        """Hungarian projection of the assignment: each letter gets exactly
        one grid value; surplus values fall to the dummy columns."""
        from scipy.optimize import linear_sum_assignment

        from .rung1_sinkhorn import sinkhorn

        cost = -sinkhorn(logits / self.tau_end).log().clamp_min(-30).numpy()
        rows, cols = linear_sum_assignment(cost)
        u = np.zeros(A, dtype=np.float64)
        for r, c in zip(rows, cols):
            if c < A:
                u[c] = grid[r] if r < len(grid) else grid[-1]
        return u

    # -- discrete phase ------------------------------------------------------

    def _hard_objective(self, v, u, char_ids, adm, language, order=2):
        """Penalized objective for integer keys: soft-σ lattice LL minus the
        frequency-KL defense (rung-2 lesson), with the decoded letter
        frequency approximated by the emission-posterior letter mass (a
        Viterbi decode per move would dominate polish cost)."""
        vt = torch.tensor(v, dtype=torch.float32)
        ut = torch.tensor(u, dtype=torch.float32)
        with torch.no_grad():
            logb = self._log_emissions(vt, ut, self.sigma_end, char_ids, adm)
            ll = float(
                self.ev.score_lattice(logb, SEG_LENGTHS, language=language, order=order)
            )
            post = torch.softmax(logb[adm], dim=-1)
            p_hat = post.mean(dim=0).clamp_min(1e-9)
            prior = torch.exp(self.ev.logT(language, 1)).float()
            kl = float((p_hat * (p_hat.log() - prior.log())).sum())
        n_letters = len(char_ids) / 4.2  # E[token len] under the length prior
        return ll - self.freq_penalty_weight * n_letters * kl, ll

    def _polish(self, v, u, char_ids, adm, language):
        """Greedy hill-climb over integer moves: u_l ± 1, v_c ± 1, and swaps
        between letters with nearby values (far swaps are reachable via the
        gradient phase; keeping the move set small keeps polish ~minutes)."""
        best, _ = self._hard_objective(v, u, char_ids, adm, language)
        n_evals = 1
        for _ in range(self.polish_rounds):
            improved = False
            moves: list[tuple[str, tuple]] = []
            moves += [("u", (l, d)) for l in range(A) for d in (-1, 1)]
            moves += [("v", (c, d)) for c in range(len(v)) for d in (-1, 1)]
            moves += [
                ("swap", (i, j))
                for i in range(A)
                for j in range(i + 1, A)
                if abs(u[i] - u[j]) <= 3
            ]
            for kind, arg in moves:
                v2, u2 = v.copy(), u.copy()
                if kind == "u":
                    u2[arg[0]] += arg[1]
                elif kind == "v":
                    v2[arg[0]] += arg[1]
                else:
                    u2[arg[0]], u2[arg[1]] = u2[arg[1]], u2[arg[0]]
                cand, _ = self._hard_objective(v2, u2, char_ids, adm, language)
                n_evals += 1
                if cand > best + 1e-6:
                    best, v, u, improved = cand, v2, u2, True
            if not improved:
                break
        return v, u, best, n_evals

    # -- driver --------------------------------------------------------------

    def solve(
        self,
        char_ids: np.ndarray,
        *,
        language: str,
        restarts: int = 3,
        splits: tuple[int, ...] = (2, 1, 3),
        polish: bool = True,
        n_sym: int = N_SYM,
    ) -> Rung4Result:
        t0 = time.time()
        ranks = infer_char_orders(char_ids, n_sym=n_sym, seed=self.seed)
        char_t = torch.from_numpy(char_ids)
        best: Rung4Result | None = None
        total_evals = 0
        short = []
        # restart schedule: order candidates first (order uncertainty is the
        # empirically dominant failure mode), then alternate splits, then
        # jittered repeats of the best-scored order.
        combos = [(k, s) for s in splits for k in range(len(ranks))]
        for r in range(restarts):
            g = torch.Generator().manual_seed(self.seed + 1000 * r)
            k, split = combos[r % len(combos)]
            rank = ranks[k]
            adm_np = admissible_mask(char_ids, rank)
            adm_t = torch.from_numpy(adm_np)
            v0 = order_derived_values(rank, split)
            grid, logits, n = self._gradient_phase(char_ids, adm_np, language, v0, g)
            total_evals += n
            v_i = np.rint(v0)
            u_i = self._project_u(grid, logits)
            if polish:
                v_i, u_i, score, n = self._polish(v_i, u_i, char_t, adm_t, language)
                total_evals += n
            else:
                score, _ = self._hard_objective(v_i, u_i, char_t, adm_t, language)
                total_evals += 1
            _, raw_ll3 = self._hard_objective(
                v_i, u_i, char_t, adm_t, language, order=3
            )
            starts, letters, _ = self.decode_with_key(
                char_ids, v_i, u_i, language=language, rank=rank
            )
            total_evals += 2
            short.append(
                (
                    v_i.astype(np.int64),
                    u_i.astype(np.int64),
                    letters,
                    float(score),
                    float(raw_ll3),
                    rank,
                )
            )
            if best is None or score > best.score:
                best = Rung4Result(
                    v=v_i.astype(np.int64),
                    u=u_i.astype(np.int64),
                    score=score,
                    raw_ll=raw_ll3,
                    decoded=letters,
                    seg_starts=starts,
                    n_evals=total_evals,
                    restarts_used=r + 1,
                    wall_s=time.time() - t0,
                    extra={"split": split, "rank": rank},
                )
        assert best is not None
        best.n_evals = total_evals
        best.wall_s = time.time() - t0
        best.shortlist = sorted(short, key=lambda x: -x[3])
        return best

    def solve_segmented(
        self,
        char_ids: np.ndarray,
        token_starts: np.ndarray,
        *,
        language: str,
        restarts: int = 3,
        splits: tuple[int, ...] = (2, 1, 3),
        polish: bool = True,
    ) -> Rung4Result:
        """Fixed-segmentation solve (task 6.1): the lattice is the observed
        token boundaries, the order prior is not needed and ``v`` is seeded
        from within-token positions. Same gradient / projection / polish
        machinery as ``solve``."""
        t0 = time.time()
        n_sym = int(char_ids.max()) + 1
        rank = positional_rank(char_ids, token_starts, n_sym=n_sym)
        adm_np = segmented_admissible_mask(char_ids, token_starts)
        if not adm_np.any():
            raise ValueError(
                "no admissible segment: token lengths outside SEG_LENGTHS?"
            )
        adm_t = torch.from_numpy(adm_np)
        char_t = torch.from_numpy(char_ids)
        best: Rung4Result | None = None
        total_evals = 0
        short = []
        for r in range(restarts):
            g = torch.Generator().manual_seed(self.seed + 1000 * r)
            split = splits[r % len(splits)]
            v0 = order_derived_values(rank, split)
            grid, logits, n = self._gradient_phase(char_ids, adm_np, language, v0, g)
            total_evals += n
            v_i = np.rint(v0)
            u_i = self._project_u(grid, logits)
            if polish:
                v_i, u_i, score, n = self._polish(v_i, u_i, char_t, adm_t, language)
                total_evals += n
            else:
                score, _ = self._hard_objective(v_i, u_i, char_t, adm_t, language)
                total_evals += 1
            _, raw_ll3 = self._hard_objective(
                v_i, u_i, char_t, adm_t, language, order=3
            )
            starts, letters, _ = self.decode_segmented(
                char_ids, adm_np, v_i, u_i, language=language
            )
            total_evals += 2
            short.append(
                (
                    v_i.astype(np.int64),
                    u_i.astype(np.int64),
                    letters,
                    float(score),
                    float(raw_ll3),
                    rank,
                )
            )
            if best is None or score > best.score:
                best = Rung4Result(
                    v=v_i.astype(np.int64),
                    u=u_i.astype(np.int64),
                    score=score,
                    raw_ll=raw_ll3,
                    decoded=letters,
                    seg_starts=starts,
                    n_evals=total_evals,
                    restarts_used=r + 1,
                    wall_s=time.time() - t0,
                    extra={"split": split, "rank": rank, "segmented": True},
                )
        assert best is not None
        best.n_evals = total_evals
        best.wall_s = time.time() - t0
        best.shortlist = sorted(short, key=lambda x: -x[3])
        return best

    def decode_segmented(self, char_ids, adm_np, v, u, *, language):
        """Viterbi decode under a given key on a fixed admissible lattice."""
        ids = torch.from_numpy(np.asarray(char_ids))
        adm = torch.from_numpy(np.asarray(adm_np))
        logb = self._log_emissions(
            torch.tensor(v, dtype=torch.float32),
            torch.tensor(u, dtype=torch.float32),
            self.sigma_end,
            ids,
            adm,
        )
        return self.ev.viterbi_lattice(logb, SEG_LENGTHS, language=language)

    def decode_with_key(
        self,
        char_ids: np.ndarray,
        v: np.ndarray,
        u: np.ndarray,
        *,
        language: str,
        rank: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Viterbi decode under a given integer key (truth-validation path)."""
        if rank is None:
            rank = infer_char_order(char_ids)
        adm = torch.from_numpy(admissible_mask(char_ids, rank))
        ids = torch.from_numpy(char_ids)
        logb = self._log_emissions(
            torch.tensor(v, dtype=torch.float32),
            torch.tensor(u, dtype=torch.float32),
            self.sigma_end,
            ids,
            adm,
        )
        return self.ev.viterbi_lattice(logb, SEG_LENGTHS, language=language)


# -- alignment metric --------------------------------------------------------


@njit(cache=True, nogil=True)
def _levenshtein(decoded, truth):
    n, m = len(decoded), len(truth)
    prev = np.arange(m + 1)
    cur = np.empty(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        cur[0] = i
        d = decoded[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (d != truth[j - 1])
            best = prev[j] + 1
            best = min(best, cur[j - 1] + 1)
            best = min(best, sub)
            cur[j] = best
        prev, cur = cur, prev
    return prev[m]


def levenshtein_ser(decoded: np.ndarray, truth: np.ndarray) -> float:
    """Edit-distance error rate — decoded length may differ from the truth
    when the inferred segmentation splits/merges tokens. Compiled (numba):
    the O(n·m) DP in Python took ~38 s on a 7.7k-letter decode and was the
    dominant term of an alternating-loop round (called twice per round by
    the metrics)."""
    n, m = len(decoded), len(truth)
    if m == 0:
        return float(n > 0)
    d = np.ascontiguousarray(np.asarray(decoded, dtype=np.int64))
    t = np.ascontiguousarray(np.asarray(truth, dtype=np.int64))
    return float(_levenshtein(d, t)) / m
