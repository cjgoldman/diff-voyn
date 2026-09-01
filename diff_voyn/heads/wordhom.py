"""Word-level homophonic head with a bigram-extended plaintext alphabet and
the repeat rule — Boxer's hypothesis proper, without the arithmetic.

Boxer's ``pseudo_vms`` (rung 4) is a key-*compression* device: the integer
sum-to-target rule lets a 16-value key name thousands of homophones. The
hypothesis he actually advances (personal communication, 2026-08) is the
cipher behind it:

* the unit of ciphertext is the **word token**;
* the plaintext alphabet is the language's letters plus its ``n`` most
  frequent DOUBLED letters as single characters (n ≈ 4–5; ``ss``, ``ll``,
  ``tt`` … — the ß analogy; clarified with the user 2026-08-24, and the
  only reading under which the text's residual doubling rate matches the
  manuscript's adjacent-token repeat rate, see ``language_targets``);
* many tokens stand for one unit (homophonic, many-to-one), with no key
  structure assumed;
* the **repeat rule** — a unit written twice in a row (``ll``, ``ss``,
  ``en·en``) is enciphered with the *same* token twice.

Dropping the arithmetic leaves a homophonic substitution over word types
with an (A + n)-letter target alphabet: exactly rung 2's problem
(:class:`~.rung2_homophonic.HomophonicHead`) with (a) targets that emit one
OR two letters and (b) the repeat rule as a constraint on the map. The
converse of the repeat rule is what the decoder can use: every doubled unit
in the plaintext appears as a repeated token, so a map under which two
*different* adjacent tokens decode to the same unit violates the rule. That
is charged as a penalty in the discrete objective (``repeat_weight`` nats
per violation; the true key of a rule-following cipher has none).

Key: ``sym_to_unit`` (n_symbols,) ints in ``0..A+n-1`` — the first ``A``
targets are letters, the rest the language's top-``n`` doubled letters (so the
target set is part of the language hypothesis, as Boxer states it).
Search: frequency init + random restarts of SA over single-symbol
reassignments on the exact pentagram objective with the rung-2
letter-frequency KL penalty and the repeat penalty, then greedy polish.
No ELBO polish: the neighbourhood (n_symbols × (A+n) moves) is far past the
scale at which ``ladder.elbo_polish`` was shown to select on noise
(``docs/phase6_status.md``, Borg), so the outer tier is shortlist rescoring
+ MDL selection only.

Identifiability caveat (the reason the arithmetic existed): a word type
seen once constrains nothing. Running the head on the top-K types only and
dropping the rest was tried first and fails on its own positive control —
at 69% token coverage the TRUE decode of the gapped stream costs 5.7
bits/char (clean text 2.9) because the deletions destroy the n-gram
context, and greedy polish then walks 56% of the symbols away from the
truth. So the head maps EVERY type (``wordtypesall``), rare types being
free parameters that are paid for in key bits (n_types · log2(A+n)) and
that polish re-fits at will (on the synthetic positive: 22% of types move
under polish from the true key, 0.6% of the top-400 occurrence-weighted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .evaluator import TokenEmission
from .ngram import A

N_BIGRAMS = 5
# unit-set spec strings: ``d<k>`` = the language's top-k doubled letters
# (Boxer's hypothesis as clarified, the default), ``d<k>b<m>`` = those plus
# the top-m NON-doubled bigrams as further single-character units (the
# 2026-08-30 "doubles + bigrams" variant; see ``language_targets``)
UNITS_DEFAULT = "d5"
_UNITS_RE = re.compile(r"^d(\d+)(?:b(\d+))?$")


def parse_units(spec: str | None) -> tuple[int, int]:
    """``"d5"`` → (5, 0); ``"d5b20"`` → (5, 20)."""
    m = _UNITS_RE.match(spec or UNITS_DEFAULT)
    if m is None:
        raise ValueError(f"bad unit-set spec {spec!r} (want d<k> or d<k>b<m>)")
    return int(m.group(1)), int(m.group(2) or 0)


def units_suffix(spec: str | None) -> str:
    """File-name suffix of a unit-set spec: '' for the default, else '_<spec>'."""
    return "" if (spec or UNITS_DEFAULT) == UNITS_DEFAULT else f"_{spec}"


# -- target alphabet ---------------------------------------------------------


@dataclass(frozen=True)
class UnitTargets:
    """Letters 0..A-1 then ``bigrams`` (list of (a, b) letter-id pairs)."""

    bigrams: tuple[tuple[int, int], ...]

    @property
    def n(self) -> int:
        return A + len(self.bigrams)

    @property
    def first(self) -> np.ndarray:
        return np.concatenate([np.arange(A), [a for a, _ in self.bigrams]]).astype(
            np.int64
        )

    @property
    def second(self) -> np.ndarray:
        return np.concatenate([np.full(A, -1), [b for _, b in self.bigrams]]).astype(
            np.int64
        )

    def as_list(self) -> list[list[int]]:
        return [[int(a), int(b)] for a, b in self.bigrams]

    @classmethod
    def from_list(cls, bigrams) -> UnitTargets:
        return cls(tuple((int(a), int(b)) for a, b in bigrams))


def language_targets(
    evaluator,
    language: str,
    n_bigrams: int = N_BIGRAMS,
    n_general: int = 0,
    *,
    units: str | None = None,
) -> UnitTargets:
    """The language's ``n_bigrams`` most frequent DOUBLED letters (aa) under
    the frozen n-gram LM, p(aa) = p1(a) p2(a|a) — the ß-like extra
    characters of the hypothesis — followed by its ``n_general`` most
    frequent NON-doubled bigrams (a≠b) when asked for (``units="d5b20"``;
    ``units`` overrides both counts). Doubles come first so a key in the
    ``d<k>`` space is a key in the ``d<k>b<m>`` space unchanged.

    (General bigrams were tried first as *the* hypothesis and rejected by
    the doubling control: with er/en/… as units the text's doubled-unit
    rate stays at 35–44 per 1000 against the manuscript's 7–9; with the
    top-5 doubled letters it is 10–16. The ``b<m>`` variant is a decoder
    over-specification study, not a claim about the manuscript.)"""
    if units is not None:
        n_bigrams, n_general = parse_units(units)
    l1 = evaluator.logT(language, 1).cpu().numpy().reshape(A)
    l2 = evaluator.logT(language, 2).cpu().numpy().reshape(A, A)
    joint = l1[:, None] + l2
    top = np.argsort(-np.diagonal(joint))[:n_bigrams]
    pairs = [(int(a), int(a)) for a in top]
    if n_general > 0:
        off = joint.copy()
        np.fill_diagonal(off, -np.inf)
        order = np.argsort(-off, axis=None)[:n_general]
        pairs += [(int(i // A), int(i % A)) for i in order]
    return UnitTargets(tuple(pairs))


def hypothesis_targets(
    evaluator, language: str, *, units: str | None = None, inst: dict | None = None
) -> UnitTargets:
    """The unit space a DECODER runs in for ``language``: an instance's
    ``truth.hyp_bigrams`` override when present (battery cells whose
    hypothesis deliberately differs from the cipher, e.g. ``revdouble``),
    else ``language_targets`` under the ``units`` spec."""
    hb = (inst or {}).get("truth", {}).get("hyp_bigrams")
    if hb is not None:
        return UnitTargets.from_list(hb)
    return language_targets(evaluator, language, units=units)


def project_key(
    true_map: np.ndarray, cipher_targets: UnitTargets, hyp_targets: UnitTargets
) -> np.ndarray:
    """The cipher's true key expressed in the hypothesis' unit space: letters
    stay, a bigram unit the hypothesis also has takes its index there, a
    bigram unit the hypothesis lacks falls back to its first letter (the
    closest representable key — one wrong letter per occurrence, measured as
    such by the letter-level SER). Identity when the spaces coincide."""
    true_map = np.asarray(true_map, dtype=np.int64)
    if cipher_targets.bigrams == hyp_targets.bigrams:
        return true_map
    idx = {pair: A + i for i, pair in enumerate(hyp_targets.bigrams)}
    lut = np.arange(cipher_targets.n, dtype=np.int64)
    for i, pair in enumerate(cipher_targets.bigrams):
        lut[A + i] = idx.get(pair, pair[0])
    return lut[true_map]


def targets_from_ids(
    ids: np.ndarray, n_bigrams: int = N_BIGRAMS, n_general: int = 0
) -> UnitTargets:
    """Top doubled letters (then top non-doubled bigrams) of a letter stream
    (out-of-inventory generators)."""
    ids = np.asarray(ids, dtype=np.int64)
    c = np.bincount(ids[:-1] * A + ids[1:], minlength=A * A).reshape(A, A)
    top = np.argsort(-np.diagonal(c))[:n_bigrams]
    pairs = [(int(a), int(a)) for a in top]
    if n_general > 0:
        off = c.astype(float)
        np.fill_diagonal(off, -1.0)
        order = np.argsort(-off, axis=None)[:n_general]
        pairs += [(int(i // A), int(i % A)) for i in order]
    return UnitTargets(tuple(pairs))


# -- unit streams ------------------------------------------------------------


def expand_units(units: np.ndarray, targets: UnitTargets) -> np.ndarray:
    """Unit ids -> letter ids (bigram units emit two letters)."""
    units = np.asarray(units, dtype=np.int64)
    first, second = targets.first[units], targets.second[units]
    isbig = second >= 0
    n_out = len(units) + int(isbig.sum())
    starts = np.concatenate([[0], np.cumsum(1 + isbig)[:-1]])
    out = np.empty(n_out, dtype=np.int64)
    out[starts] = first
    out[starts[isbig] + 1] = second[isbig]
    return out


def segment_units(plain: np.ndarray, targets: UnitTargets) -> np.ndarray:
    """Greedy left-to-right segmentation of a letter stream into units: a
    top bigram at the cursor is taken as one unit, else the letter."""
    plain = np.asarray(plain, dtype=np.int64)
    code = {a * A + b: A + i for i, (a, b) in enumerate(targets.bigrams)}
    out = []
    i = 0
    n = len(plain)
    while i < n:
        if i + 1 < n and (plain[i] * A + plain[i + 1]) in code:
            out.append(code[plain[i] * A + plain[i + 1]])
            i += 2
        else:
            out.append(int(plain[i]))
            i += 1
    return np.asarray(out, dtype=np.int64)


def doubling_rate(units: np.ndarray) -> float:
    """Fraction of adjacent unit pairs that are identical — the rate at
    which a rule-following cipher must show a repeated token."""
    units = np.asarray(units)
    return float(np.mean(units[1:] == units[:-1])) if len(units) > 1 else 0.0


# -- repeat-rule bookkeeping -------------------------------------------------


def adjacency(symbols: np.ndarray, token_pos: np.ndarray | None) -> np.ndarray:
    """(n-1,) True where covered positions i, i+1 were adjacent tokens in
    the original stream (no uncovered token between them)."""
    n = len(symbols)
    if token_pos is None:
        return np.ones(max(n - 1, 0), dtype=bool)
    tp = np.asarray(token_pos, dtype=np.int64)
    return tp[1:] == tp[:-1] + 1


def repeat_positions(symbols: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """(n,) True at position i+1 when token i+1 repeats token i (adjacent)."""
    symbols = np.asarray(symbols)
    out = np.zeros(len(symbols), dtype=bool)
    if len(symbols) > 1:
        out[1:] = adj & (symbols[1:] == symbols[:-1])
    return out


def rule_violations(units: np.ndarray, symbols: np.ndarray, adj: np.ndarray) -> int:
    """Adjacent, DIFFERENT tokens decoding to the same unit — impossible
    under the repeat rule."""
    if len(units) < 2:
        return 0
    same_unit = units[1:] == units[:-1]
    diff_sym = symbols[1:] != symbols[:-1]
    return int((adj & same_unit & diff_sym).sum())


def choice_bits_total(
    sym_to_unit: np.ndarray,
    symbols: np.ndarray,
    repeats: np.ndarray,
    n_targets: int,
) -> float:
    """Bits to name the emitted token given plaintext + key: uniform among
    the unit's homophones, and zero at repeat positions (forced by the
    rule)."""
    m = np.asarray(sym_to_unit, dtype=np.int64)
    n_hom = np.bincount(m, minlength=n_targets)
    per = np.log2(np.maximum(n_hom[m[symbols]], 1))
    per[repeats] = 0.0
    return float(per.sum())


# -- head --------------------------------------------------------------------


@dataclass
class WordHomResult:
    sym_to_unit: np.ndarray
    hard_score: float
    n_evals: int
    restarts_used: int
    raw_ll: float = float("nan")
    violations: int = 0
    # [(map, penalized_score, raw_ll, source)] best first, distinct maps
    shortlist: list = field(default_factory=list)


class WordHomophonicHead:
    def __init__(
        self,
        evaluator,
        *,
        targets: UnitTargets | None = None,
        n_bigrams: int = N_BIGRAMS,
        units: str | None = None,
        rescore_order: int = 5,
        freq_penalty_weight: float = 1.0,
        repeat_weight: float = 4.0,
        seed: int = 0,
    ):
        self.ev = evaluator
        self.targets = targets
        self.n_bigrams = n_bigrams
        # unit-set spec (``parse_units``); overrides ``n_bigrams`` when given
        self.units = units
        self.rescore_order = rescore_order
        self.freq_penalty_weight = freq_penalty_weight
        self.repeat_weight = repeat_weight
        self.seed = seed
        self._prior: dict[str, np.ndarray] = {}
        # optional (n_symbols,) bool: "wildcard" types whose letters are
        # charged a constant and reset the n-gram context (hapax study);
        # frozen out of SA/polish proposals
        self.wild_types: np.ndarray | None = None

    def targets_for(self, language: str) -> UnitTargets:
        if self.targets is not None:
            return self.targets
        if self.units is not None:
            return language_targets(self.ev, language, units=self.units)
        return language_targets(self.ev, language, self.n_bigrams)

    def _log_prior(self, language: str) -> np.ndarray:
        if language not in self._prior:
            self._prior[language] = self.ev.logT(language, 1).cpu().numpy().ravel()
        return self._prior[language]

    # -- objective -----------------------------------------------------------

    def objective(
        self,
        sym_map: np.ndarray,
        symbols: np.ndarray,
        adj: np.ndarray,
        language: str,
        targets: UnitTargets,
    ) -> float:
        if self.wild_types is not None:
            from .wordhom_state import WordHomObjectiveState

            return float(
                WordHomObjectiveState(
                    self, symbols, adj, sym_map, language, targets
                ).score
            )
        units = sym_map[symbols]
        letters = expand_units(units, targets)
        ll = self.ev.score_hard(letters, language=language, order=self.rescore_order)
        emp = np.bincount(letters, minlength=A) / len(letters)
        nz = emp > 0
        kl = float((emp[nz] * (np.log(emp[nz]) - self._log_prior(language)[nz])).sum())
        viol = rule_violations(units, symbols, adj)
        return (
            ll
            - self.freq_penalty_weight * len(letters) * kl
            - self.repeat_weight * viol
        )

    # -- inits ---------------------------------------------------------------

    def frequency_init(
        self, symbols: np.ndarray, n_symbols: int, language: str, targets: UnitTargets
    ) -> np.ndarray:
        """Heaviest symbols to the units whose frequency quota is least
        filled (rung 2's homophone-flattening inversion, over units)."""
        occ = np.bincount(symbols, minlength=n_symbols)
        l1 = self._log_prior(language)
        l2 = self.ev.logT(language, 2).cpu().numpy().reshape(A, A)
        p_letters = np.exp(l1)
        p_big = np.array([np.exp(l1[a] + l2[a, b]) for a, b in targets.bigrams])
        prior = np.concatenate([p_letters, p_big])
        # letters lose the mass absorbed by bigram units (approximately)
        for i, (a, b) in enumerate(targets.bigrams):
            prior[a] -= p_big[i] / 2
            prior[b] -= p_big[i] / 2
        prior = np.maximum(prior, 1e-4)
        quota = prior / prior.sum() * occ.sum()
        filled = np.zeros(targets.n)
        sym_map = np.zeros(n_symbols, dtype=np.int64)
        for s in np.argsort(-occ):
            u = int(np.argmax(quota - filled))
            sym_map[s] = u
            filled[u] += occ[s]
        return sym_map

    # -- search --------------------------------------------------------------

    def sa_phase(
        self,
        symbols: np.ndarray,
        adj: np.ndarray,
        sym_map: np.ndarray,
        language: str,
        targets: UnitTargets,
        rng: np.random.Generator,
        steps: int = 100_000,
        t_start: float = 15.0,
        t_end: float = 0.5,
    ) -> tuple[np.ndarray, float, int]:
        """SA over single-symbol reassignments (occurrence-weighted
        proposals so frequent types are visited often) + greedy polish;
        returns the best map visited."""
        from .wordhom_state import WordHomObjectiveState

        n_symbols = len(sym_map)
        occ = np.bincount(symbols, minlength=n_symbols).astype(float)
        p_prop = occ + 1.0
        if self.wild_types is not None:
            p_prop[np.asarray(self.wild_types, dtype=bool)] = 0.0
        p_prop = p_prop / p_prop.sum()
        cdf = np.cumsum(p_prop)
        cdf[-1] = 1.0
        st = WordHomObjectiveState(self, symbols, adj, sym_map, language, targets)
        best_map, _, n_evals = st.sa(rng, steps, t_start, t_end, cdf)
        sym_map, score, n2 = self.polish(symbols, adj, best_map, language, targets)
        return sym_map, score, n_evals + 1 + n2

    def polish(
        self,
        symbols: np.ndarray,
        adj: np.ndarray,
        sym_map: np.ndarray,
        language: str,
        targets: UnitTargets,
        max_sweeps: int = 20,
    ) -> tuple[np.ndarray, float, int]:
        """Greedy single-symbol reassignment sweeps until no move improves
        (best-improvement per symbol, deltas from the incremental state).
        The returned score is the full objective recomputed once, so a
        caller comparing it against ``objective()`` sees no drift."""
        from .wordhom_state import WordHomObjectiveState

        st = WordHomObjectiveState(self, symbols, adj, sym_map, language, targets)
        _, n_evals = st.polish(max_sweeps)
        sym_map = st.sym_map.copy()
        score = self.objective(sym_map, symbols, adj, language, targets)
        return sym_map, score, n_evals + 1

    # -- EM initializer (Baum-Welch decipherment, rung-3 lineage) ------------

    def _emissions(self, symbols: np.ndarray, q: torch.Tensor, targets: UnitTargets):
        """Per-token branch structure from a (n_symbols, n_targets) prob
        matrix: a one-letter branch carrying the letter mass of the row and
        one two-letter branch per doubled unit. Multilinear in ``q`` (each
        emitted letter multiplies one entry), so expected counts are
        ``q * dLL/dq``."""
        eye = torch.eye(A)
        dbl = [(A + i, eye[a], eye[b]) for i, (a, b) in enumerate(targets.bigrams)]
        qs = q[torch.from_numpy(np.asarray(symbols, dtype=np.int64))]
        w_uni = qs[:, :A].sum(1)
        dist_uni = qs[:, :A] / w_uni.clamp_min(1e-30)[:, None]
        log_w_uni = torch.log(w_uni.clamp_min(1e-30))
        log_dbl = torch.log(qs[:, A:].clamp_min(1e-30))
        ems = []
        for i in range(len(symbols)):
            branches = [(log_w_uni[i], [dist_uni[i]])]
            for j, (_, ea, eb) in enumerate(dbl):
                branches.append((log_dbl[i, j], [ea, eb]))
            ems.append(TokenEmission(branches=branches))
        return ems

    def em_phase(
        self,
        symbols: np.ndarray,
        n_symbols: int,
        language: str,
        targets: UnitTargets,
        rng: np.random.Generator,
        iters: int = 40,
        alpha: float = 0.05,
        t_start: float = 2.0,
        chunk_tokens: int = 2000,
        noise: float = 0.3,
    ) -> tuple[np.ndarray, torch.Tensor, int]:
        """Exact EM over the trigram-state segmental DP with deterministic
        annealing (rung 3's ``_em_phase``), all types jointly; returns the
        argmax map, the final posteriors and the number of DP passes.
        Rows start at the language prior over units (letters ∝ p1, doubled
        units ∝ p1(a) p2(a|a)) with per-restart noise."""
        l1 = self._log_prior(language)
        l2 = self.ev.logT(language, 2).cpu().numpy().reshape(A, A)
        log_prior = np.concatenate([l1, [l1[a] + l2[a, b] for a, b in targets.bigrams]])
        log_prior = torch.from_numpy(log_prior).float()
        logits = (
            log_prior[None, :]
            + noise
            * torch.from_numpy(rng.standard_normal((n_symbols, targets.n))).float()
        )
        probs = torch.softmax(logits, 1)
        symbols = np.asarray(symbols, dtype=np.int64)
        n_chunks = max(1, (len(symbols) + chunk_tokens - 1) // chunk_tokens)
        n_evals = 0
        for it in range(iters):
            leaf = probs.clone().requires_grad_(True)
            ll_total = torch.zeros(())
            for ci in range(n_chunks):
                chunk = symbols[ci * chunk_tokens : (ci + 1) * chunk_tokens]
                ems = self._emissions(chunk, leaf, targets)
                ll_total = ll_total + self.ev.score_segmental(ems, language=language)
                n_evals += 1
            ll_total.backward()
            temp = t_start + (1.0 - t_start) * min(1.0, it / max(iters * 0.6, 1))
            with torch.no_grad():
                counts = (leaf * leaf.grad).clamp_min(0.0) + alpha
                counts = counts.pow(1.0 / temp)
                probs = (counts / counts.sum(1, keepdim=True)).detach()
        return probs.argmax(1).numpy().astype(np.int64), probs, n_evals

    def solve(
        self,
        symbols: np.ndarray,
        n_symbols: int,
        *,
        language: str,
        token_pos: np.ndarray | None = None,
        restarts: int = 16,
        sa_steps: int = 100_000,
        t_start: float = 15.0,
        t_end: float = 0.5,
        shortlist: int = 8,
    ) -> WordHomResult:
        symbols = np.asarray(symbols, dtype=np.int64)
        adj = adjacency(symbols, token_pos)
        targets = self.targets_for(language)
        rng = np.random.default_rng(self.seed)
        found = []
        total_evals = 0
        for r in range(restarts):
            init = (
                self.frequency_init(symbols, n_symbols, language, targets)
                if r == 0
                else rng.integers(0, targets.n, size=n_symbols)
            )
            m, s, n = self.sa_phase(
                symbols,
                adj,
                np.asarray(init, dtype=np.int64).copy(),
                language,
                targets,
                rng,
                steps=sa_steps,
                t_start=t_start,
                t_end=t_end,
            )
            total_evals += n
            found.append((m, float(s), f"sa{r}" if r else "freq+sa0"))
        found.sort(key=lambda x: -x[1])
        seen = set()
        short = []
        for m, s, src in found:
            k = m.tobytes()
            if k in seen:
                continue
            seen.add(k)
            raw = self.ev.score_hard(
                expand_units(m[symbols], targets),
                language=language,
                order=self.rescore_order,
            )
            short.append((m, s, float(raw), src))
            if len(short) >= shortlist:
                break
        best_m, best_s, best_raw, _ = short[0]
        return WordHomResult(
            best_m,
            best_s,
            total_evals,
            restarts,
            raw_ll=best_raw,
            violations=rule_violations(best_m[symbols], symbols, adj),
            shortlist=short,
        )


# -- synthetic cipher (controls) ---------------------------------------------


@dataclass
class WordHomCipher:
    """Word-level homophonic cipher over ``targets`` with the repeat rule.

    ``n_types`` homophones are allocated across the units present in the
    plaintext proportionally to unit frequency (≥ 1 each); within a unit
    the token is drawn Zipf(``zipf_s``) over its homophones (so a long tail
    of rare types appears, as in the manuscript); a unit equal to the
    previous unit re-uses the previous token.
    """

    targets: UnitTargets
    n_types: int = 2500
    zipf_s: float = 1.0

    def encipher(self, plain: np.ndarray, rng: np.random.Generator):
        units = segment_units(plain, self.targets)
        counts = np.bincount(units, minlength=self.targets.n).astype(float)
        present = counts > 0
        alloc = np.zeros(self.targets.n, dtype=int)
        alloc[present] = 1
        extra = self.n_types - int(present.sum())
        if extra > 0:
            share = counts / counts.sum() * extra
            add = np.floor(share).astype(int)
            rem = extra - int(add.sum())
            frac = share - add
            for i in np.argsort(-frac)[:rem]:
                add[i] += 1
            alloc += add
        homs = {}
        nxt = 0
        for u in range(self.targets.n):
            homs[u] = np.arange(nxt, nxt + alloc[u])
            nxt += alloc[u]
        weights = {}
        for u, h in homs.items():
            if len(h):
                w = np.arange(1, len(h) + 1, dtype=float) ** (-self.zipf_s)
                weights[u] = w / w.sum()
        tokens = np.empty(len(units), dtype=np.int64)
        for i, u in enumerate(units):
            if i > 0 and u == units[i - 1]:
                tokens[i] = tokens[i - 1]
            else:
                tokens[i] = rng.choice(homs[u], p=weights[u])
        sym_to_unit = np.empty(nxt, dtype=np.int64)
        for u, h in homs.items():
            sym_to_unit[h] = u
        return units, tokens, sym_to_unit


def unit_ser(decoded_letters: np.ndarray, truth_letters: np.ndarray) -> float:
    """Letter-level edit-distance SER between the expanded decode and the
    plaintext (decodes may change length when a bigram unit is mis-mapped)."""
    from .rung4_arithmetic import levenshtein_ser

    return levenshtein_ser(
        np.asarray(decoded_letters, dtype=np.int64),
        np.asarray(truth_letters, dtype=np.int64),
    )
