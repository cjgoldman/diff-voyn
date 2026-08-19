"""Rung 3 — Naibbe mixed unigram-bigram head — task CH.6 (= 5.4).

Productionizes the inverse-note skeleton against the pinned tables: three
soft inverse maps ``U`` (unigram type -> letter), ``Pre``, ``Suf`` (softmax
over the 23-letter Naibbe support inside the frozen 25-letter alphabet),
plus a global unigram-vs-bigram mixture logit. Tokens are pre-parsed once by
:class:`~diff_voyn.heads.naibbe_parse.NaibbeParser` (the exact structural
prior); the frozen evaluator's semi-Markov DP marginalizes segmentation and
soft letter identity jointly (``score_segmental``), so the only latent
per-token variable is which parse branch produced it.

Search: Adam on the ~9.5k logits with row-entropy annealing, then argmax
maps; optional discrete single-type reassignment polish under the hard DP.
Restart-friendly by seed. The head never names its evaluator (CH.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .evaluator import TokenEmission
from .naibbe_parse import NaibbeParser, TokenParse
from .ngram import A

NEG_BIG = -1e9  # soft -inf: keeps grads finite (never a true -inf blend)

STATES = ("unigram", "prefix", "suffix")


@dataclass
class Rung3Result:
    uni_map: np.ndarray  # (n_uni,) type -> letter idx (frozen alphabet)
    pre_map: np.ndarray
    suf_map: np.ndarray
    score: float
    n_evals: int
    restarts_used: int

    def map_accuracy(self, parser: NaibbeParser, weights=None) -> dict[str, float]:
        out = {}
        for state, got in (
            ("unigram", self.uni_map),
            ("prefix", self.pre_map),
            ("suffix", self.suf_map),
        ):
            out[state] = float(np.mean(got == parser.truth[state]))
        return out


class NaibbeHead:
    def __init__(
        self,
        evaluator,
        parser: NaibbeParser,
        *,
        steps: int = 400,
        lr: float = 0.1,
        entropy_weight: float = 0.1,
        chunk_tokens: int = 512,
        seed: int = 0,
    ):
        self.ev = evaluator
        self.parser = parser
        self.steps = steps
        self.lr = lr
        self.entropy_weight = entropy_weight
        self.chunk_tokens = chunk_tokens
        self.seed = seed
        self.support = torch.from_numpy(parser.letter_support)

    # -- soft maps -----------------------------------------------------------

    def _rows(self, logits: torch.Tensor) -> torch.Tensor:
        """(n, 23) logits -> (n, A) distributions supported on Naibbe letters."""
        q = logits.new_zeros(logits.shape[0], A)
        q[:, self.support] = torch.softmax(logits, dim=1)
        return q

    def _emissions(
        self,
        parses: list[TokenParse],
        qu: torch.Tensor,
        qp: torch.Tensor,
        qs: torch.Tensor,
        w_logit: torch.Tensor,
    ) -> list[TokenEmission]:
        log_w_uni = torch.nn.functional.logsigmoid(w_logit)
        log_w_bi = torch.nn.functional.logsigmoid(-w_logit)
        ems = []
        for p in parses:
            branches = []
            if p.uni is not None:
                w = log_w_uni if p.bi else torch.zeros(())
                branches.append((w, [qu[p.uni]]))
            if p.bi:
                w = log_w_bi if p.uni is not None else torch.zeros(())
                w = w - float(np.log(len(p.bi)))
                for pre_id, suf_id in p.bi:
                    branches.append((w, [qp[pre_id], qs[suf_id]]))
            ems.append(TokenEmission(branches=branches))
        return ems

    # -- solving -------------------------------------------------------------

    def _gradient_phase(
        self, parses: list[TokenParse], language: str, g: torch.Generator
    ) -> tuple[dict[str, torch.Tensor], int]:
        n_sup = len(self.parser.letter_support)
        params = {
            "U": 0.1 * torch.randn(self.parser.n_uni, n_sup, generator=g),
            "Pre": 0.1 * torch.randn(self.parser.n_pre, n_sup, generator=g),
            "Suf": 0.1 * torch.randn(self.parser.n_suf, n_sup, generator=g),
            "w": torch.zeros(()),
        }
        for p in params.values():
            p.requires_grad_(True)
        opt = torch.optim.Adam(params.values(), lr=self.lr)
        n_chunks = max(1, (len(parses) + self.chunk_tokens - 1) // self.chunk_tokens)
        n_evals = 0
        for step in range(self.steps):
            frac = step / max(self.steps - 1, 1)
            # one random chunk per step (SGD over the token stream)
            ci = int(torch.randint(n_chunks, (1,), generator=g))
            chunk = parses[ci * self.chunk_tokens : (ci + 1) * self.chunk_tokens]
            qu, qp, qs = (
                self._rows(params["U"]),
                self._rows(params["Pre"]),
                self._rows(params["Suf"]),
            )
            ems = self._emissions(chunk, qu, qp, qs, params["w"])
            ll = self.ev.score_segmental(ems, language=language)
            n_letters = sum(2 - (p.uni is not None) for p in chunk)  # rough
            ent = sum(
                -(q * torch.log(q.clamp_min(1e-9))).sum(1).mean() for q in (qu, qp, qs)
            )
            loss = -ll / max(n_letters, 1) + self.entropy_weight * frac * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_evals += 1
        return {k: v.detach() for k, v in params.items()}, n_evals

    def _em_phase(
        self,
        parses: list[TokenParse],
        language: str,
        g: torch.Generator,
        iters: int = 50,
        alpha: float = 0.05,
        t_start: float = 2.0,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Exact EM with deterministic annealing.

        The DP likelihood is multilinear in the emission probabilities (each
        emitted letter multiplies one q[type, letter] entry), so the E-step's
        expected counts are ``q * dLL/dq`` — one autograd backward pass over
        the full token stream. M-step: normalize counts (+ Dirichlet alpha);
        annealing flattens counts early (T: t_start -> 1) to delay premature
        one-hot collapse. Far stronger per iteration than SGD on this model
        (this is classical Baum-Welch decipherment, Knight lineage).
        """
        n_sup = len(self.parser.letter_support)
        # Prior-informed init (classical Baum-Welch decipherment): rows start
        # near the language's unigram prior over the support letters, with
        # per-restart noise. A uniform-random init strands EM in far basins.
        log_prior = self.ev.logT(language, 1)[self.support].to(torch.float32)
        log_prior = log_prior - torch.logsumexp(log_prior, 0)
        probs = {
            k: torch.softmax(
                log_prior[None, :] + 0.3 * torch.randn(n, n_sup, generator=g), 1
            )
            for k, n in (
                ("U", self.parser.n_uni),
                ("Pre", self.parser.n_pre),
                ("Suf", self.parser.n_suf),
            )
        }
        w = torch.zeros(())
        n_chunks = max(1, (len(parses) + self.chunk_tokens - 1) // self.chunk_tokens)
        n_evals = 0
        for it in range(iters):
            leaves = {k: v.clone().requires_grad_(True) for k, v in probs.items()}
            w_leaf = w.clone().requires_grad_(True)
            full = {k: self._pad_support(v) for k, v in leaves.items()}
            ll_total = torch.zeros(())
            for ci in range(n_chunks):
                chunk = parses[ci * self.chunk_tokens : (ci + 1) * self.chunk_tokens]
                ems = self._emissions(
                    chunk, full["U"], full["Pre"], full["Suf"], w_leaf
                )
                ll_total = ll_total + self.ev.score_segmental(ems, language=language)
                n_evals += 1
            ll_total.backward()
            temp = t_start + (1.0 - t_start) * min(1.0, it / max(iters * 0.6, 1))
            for k in probs:
                with torch.no_grad():
                    counts = (leaves[k] * leaves[k].grad).clamp_min(0.0) + alpha
                    counts = counts.pow(1.0 / temp)
                    probs[k] = (counts / counts.sum(dim=1, keepdim=True)).detach()
            # w update: sigmoid(w) is the uni-branch prior; its expected count
            # ratio comes through the logit's gradient (d logsigmoid terms).
            with torch.no_grad():
                w = w + 0.5 * w_leaf.grad.sign() * min(1.0, abs(float(w_leaf.grad)))
        params = {k: torch.log(v.clamp_min(1e-9)) for k, v in probs.items()}
        params["w"] = w
        return params, n_evals

    def _pad_support(self, q_sup: torch.Tensor) -> torch.Tensor:
        """(n, 23) support-space probs -> (n, A) full-alphabet rows."""
        out = q_sup.new_zeros(q_sup.shape[0], A)
        out[:, self.support] = q_sup
        return out

    def _argmax_maps(self, params: dict[str, torch.Tensor]) -> tuple[np.ndarray, ...]:
        sup = self.parser.letter_support
        return tuple(sup[params[k].argmax(dim=1).numpy()] for k in ("U", "Pre", "Suf"))

    def _hard_score(
        self,
        parses: list[TokenParse],
        maps: tuple[np.ndarray, np.ndarray, np.ndarray],
        language: str,
    ) -> float:
        """Exact DP score of hard (one-hot) maps — the discrete-move scorer."""
        uni_m, pre_m, suf_m = maps
        eye = torch.eye(A)
        qu, qp, qs = eye[uni_m], eye[pre_m], eye[suf_m]
        with torch.no_grad():
            ems = self._emissions(parses, qu, qp, qs, torch.zeros(()))
            return float(self.ev.score_segmental(ems, language=language))

    def solve(
        self,
        tokens: list[str],
        *,
        language: str,
        restarts: int = 3,
        method: str = "em",
        em_iters: int = 50,
    ) -> Rung3Result:
        parses = self.parser.parse_stream(tokens)
        best: Rung3Result | None = None
        total_evals = 0
        for r in range(restarts):
            g = torch.Generator().manual_seed(self.seed + 1000 * r)
            if method == "em":
                params, n = self._em_phase(parses, language, g, iters=em_iters)
            else:
                params, n = self._gradient_phase(parses, language, g)
            total_evals += n
            maps = self._argmax_maps(params)
            score = self._hard_score(parses, maps, language)
            total_evals += 1
            if best is None or score > best.score:
                best = Rung3Result(*maps, score, total_evals, r + 1)
        assert best is not None
        best.n_evals = total_evals
        return best


# ---------------------------------------------------------------------------
# v2: block-Sinkhorn parameterization


@dataclass
class BlockResult:
    """Hard letter maps per (state, table) block + accuracy hooks."""

    block_maps: dict  # (state, table) -> (23,) row -> support-letter idx
    score: float
    n_evals: int
    restarts_used: int

    def code_accuracy(self, parser: NaibbeParser) -> dict[str, float]:
        parser.build_blocks()
        out = {}
        for state in STATES:
            got, want = [], []
            for table in NaibbeParser.TABLES:
                got.append(self.block_maps[(state, table)])
                want.append(parser.block_truth[(state, table)])
            out[state] = float(np.mean(np.concatenate(got) == np.concatenate(want)))
        out["all"] = float(
            np.mean(
                np.concatenate([self.block_maps[k] for k in self.block_maps])
                == np.concatenate([parser.block_truth[k] for k in self.block_maps])
            )
        )
        return out


class NaibbeBlockHead:
    """Rung-3 head v2: the key of each (state, table) block is a 23x23
    BIJECTION between glyph codes and letters (the published apparatus fixes
    which codes live in which block, and the deck weights). Parameterize each
    block ALICE-style with a Sinkhorn matrix — 18 blocks, ~9.5k logits but
    vastly fewer effective degrees of freedom than the free-categorical v1
    head (which plateaus at ~20% map accuracy; the bijection prior is the
    identifiability the cipher's homophony destroys at the type level).

    Emission likelihood of glyph type i (state s) at letter c:
    ``b_i(c) = sum_cells w_table * P_(s,t)[row, c]`` — with doubly-stochastic
    P this is a proper P(glyph | letter) within the block mixture, so the
    evaluator's semi-Markov advance consumes it directly (it never requires
    normalized rows).
    """

    def __init__(
        self,
        evaluator,
        parser: NaibbeParser,
        *,
        steps: int = 250,
        lr: float = 0.3,
        tau_start: float = 1.0,
        tau_end: float = 0.2,
        gumbel_scale: float = 0.3,
        chunk_tokens: int = 512,
        use_78_deck: bool = False,
        seed: int = 0,
    ):
        parser.build_blocks()
        self.ev = evaluator
        self.parser = parser
        self.steps = steps
        self.lr = lr
        self.tau_start, self.tau_end = tau_start, tau_end
        self.gumbel_scale = gumbel_scale
        self.chunk_tokens = chunk_tokens
        self.seed = seed
        self.support = torch.from_numpy(parser.letter_support)
        w = NaibbeParser.CARD_WEIGHTS[use_78_deck]
        total = sum(w.values())
        self.table_w = {t: w[t] / total for t in NaibbeParser.TABLES}
        # per state: index tensors mapping block rows -> type ids
        self.row_to_type: dict[tuple[str, str], torch.Tensor] = {}
        for state in STATES:
            ids = {g: i for i, g in enumerate(parser.types[state])}
            for table in NaibbeParser.TABLES:
                self.row_to_type[(state, table)] = torch.tensor(
                    [ids[g] for g in parser.block_codes[(state, table)]]
                )

    def _type_likelihoods(
        self, blocks: dict[tuple[str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Assemble per-type emission likelihood rows (n_types, A)."""
        out = {}
        n_types = {
            "unigram": self.parser.n_uni,
            "prefix": self.parser.n_pre,
            "suffix": self.parser.n_suf,
        }
        for state in STATES:
            q_sup = None
            for table in NaibbeParser.TABLES:
                p = blocks[(state, table)] * self.table_w[table]
                if q_sup is None:
                    q_sup = p.new_zeros(n_types[state], p.shape[1])
                q_sup = q_sup.index_add(0, self.row_to_type[(state, table)], p)
            q = q_sup.new_zeros(n_types[state], A)
            q[:, self.support] = q_sup
            out[state] = q
        return out

    def _emissions_from(self, parses, likes, w_logit):
        log_w_uni = torch.nn.functional.logsigmoid(w_logit)
        log_w_bi = torch.nn.functional.logsigmoid(-w_logit)
        ems = []
        for p in parses:
            branches = []
            if p.uni is not None:
                w = log_w_uni if p.bi else torch.zeros(())
                branches.append((w, [likes["unigram"][p.uni]]))
            if p.bi:
                w = log_w_bi if p.uni is not None else torch.zeros(())
                for pre_id, suf_id in p.bi:
                    branches.append(
                        (w, [likes["prefix"][pre_id], likes["suffix"][suf_id]])
                    )
            ems.append(TokenEmission(branches=branches))
        return ems

    def _gradient_phase(self, parses, language, g):
        from .rung1_sinkhorn import sinkhorn

        n_sup = len(self.parser.letter_support)
        logits = {
            key: (0.1 * torch.randn(n_sup, n_sup, generator=g)).requires_grad_(True)
            for key in self.row_to_type
        }
        w_logit = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([*logits.values(), w_logit], lr=self.lr)
        n_chunks = max(1, (len(parses) + self.chunk_tokens - 1) // self.chunk_tokens)
        n_evals = 0
        for step in range(self.steps):
            frac = step / max(self.steps - 1, 1)
            tau = self.tau_start * (self.tau_end / self.tau_start) ** frac
            blocks = {}
            for key, lg in logits.items():
                noise = -torch.log(
                    -torch.log(torch.rand(n_sup, n_sup, generator=g).clamp_min(1e-20))
                )
                blocks[key] = sinkhorn(
                    (lg + self.gumbel_scale * (1 - frac) * noise) / tau
                )
            likes = self._type_likelihoods(blocks)
            ci = int(torch.randint(n_chunks, (1,), generator=g))
            chunk = parses[ci * self.chunk_tokens : (ci + 1) * self.chunk_tokens]
            ems = self._emissions_from(chunk, likes, w_logit)
            ll = self.ev.score_segmental(ems, language=language)
            n_letters = sum(2 - (p.uni is not None) for p in chunk)
            loss = -ll / max(n_letters, 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_evals += 1
        return (
            {k: v.detach() for k, v in logits.items()},
            float(w_logit.detach()),
            n_evals,
        )

    def _project(self, logits):
        from scipy.optimize import linear_sum_assignment

        from .rung1_sinkhorn import sinkhorn

        maps = {}
        for key, lg in logits.items():
            cost = -sinkhorn(lg / self.tau_end).log().clamp_min(-30)
            rows, cols = linear_sum_assignment(cost.numpy())
            m = np.empty(lg.shape[0], dtype=np.int64)
            m[rows] = cols
            maps[key] = m
        return maps

    def _hard_score(self, parses, maps, language):
        blocks = {}
        for key, m in maps.items():
            b = torch.zeros(len(m), len(m))
            b[torch.arange(len(m)), torch.from_numpy(m)] = 1.0
            blocks[key] = b
        likes = self._type_likelihoods(blocks)
        with torch.no_grad():
            ems = self._emissions_from(parses, likes, torch.zeros(()))
            return float(self.ev.score_segmental(ems, language=language))

    def solve(
        self,
        tokens: list[str],
        *,
        language: str,
        restarts: int = 3,
    ) -> BlockResult:
        parses = self.parser.parse_stream(tokens)
        best = None
        total_evals = 0
        for r in range(restarts):
            g = torch.Generator().manual_seed(self.seed + 1000 * r)
            logits, _w, n = self._gradient_phase(parses, language, g)
            total_evals += n
            maps = self._project(logits)
            score = self._hard_score(parses, maps, language)
            total_evals += 1
            if best is None or score > best.score:
                best = BlockResult(maps, score, total_evals, r + 1)
        assert best is not None
        best.n_evals = total_evals
        return best
