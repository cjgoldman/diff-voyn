"""Exact incremental form of :meth:`WordHomophonicHead.objective`.

The full objective re-scores the whole letter stream (~11k–24k letters on
the manuscript windows) for every single-symbol move, although a move
touches only the occurrences of one word type (median 1; 74 % of the types
are singletons). A pentagram conditional depends on the 4 letters before
the target, so changing the letters of token ``i`` changes the conditionals
of the letters of tokens ``i..i+4`` only (every token emits ≥ 1 letter, so
4 tokens always cover ≥ 4 letters — a superset whose extra positions are
identical under the old and the new key and cancel). The state keeps

* ``tokll[t]`` — the current conditional log-prob sum of the letters of
  token ``t``, so the OLD side of a delta is a lookup and only the NEW side
  is computed, on the affected tokens;
* the letter counts as ``S1 = Σ c ln c`` and ``S2 = Σ c ln prior`` so the
  frequency-KL term is O(letters changed);
* the repeat-rule violation count, re-examined on the pairs adjacent to
  the occurrences.

The new side is one numba kernel (:func:`_scan`): for every occurrence
``p`` of a changed symbol it rebuilds the letters of tokens ``p-4 .. hi``
with ``hi = min(p+4, next occurrence-1)`` (so overlapping occurrences never
charge a token twice) and scores the letters of tokens ``p..hi`` with the
same warm-up rule as ``NgramLM.score_ids`` (order ``min(q+1, 5)`` at
absolute position ``q`` — exact for tokens 0..3 too). The delta matches
the full objective up to the float32 summation order of the full scorer
(``tests/test_wordhom.py``).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .ngram import A

CTX = 4  # pentagram context length


@njit(cache=True, nogil=True)
def _scan(
    pos,
    n_pos,
    ch_sym,
    ch_unit,
    symbols,
    units,
    first,
    second,
    tabs,
    tab_off,
    tokll,
    n_tok,
    adj,
    wild,
    out_tok,
    out_val,
):
    """Returns ``(d_ll, d_viol, n_out)``; ``out_tok[:n_out]`` /
    ``out_val[:n_out]`` are the affected tokens and their new conditional
    sums. ``tabs`` holds the flattened log-prob tables of orders 1..5
    (``tabs[tab_off[k] + ctx * A + letter]``) with ``ctx`` the base-A code
    of the ``k-1`` context letters."""
    d_ll = 0.0
    d_viol = 0
    n_out = 0
    n_ch = ch_sym.shape[0]
    buf = np.empty(2 * (2 * CTX + 1), np.int64)
    wbuf = np.zeros(2 * (2 * CTX + 1), np.int64)
    start = np.empty(2 * CTX + 2, np.int64)  # letter index where token k starts
    for j in range(n_pos):
        p = pos[j]
        nxt = pos[j + 1] if j + 1 < n_pos else n_tok
        hi = p + CTX
        hi = min(hi, nxt - 1)
        hi = min(hi, n_tok - 1)
        lo = p - CTX
        lo = max(lo, 0)
        # letters of tokens lo..hi under the new key
        n = 0
        for t in range(lo, hi + 1):
            u = units[t]
            s = symbols[t]
            for c in range(n_ch):
                if ch_sym[c] == s:
                    u = ch_unit[c]
                    break
            start[t - lo] = n
            buf[n] = first[u]
            wbuf[n] = wild[t]
            n += 1
            if second[u] >= 0:
                buf[n] = second[u]
                wbuf[n] = wild[t]
                n += 1
        start[hi - lo + 1] = n
        for t in range(p, hi + 1):
            tot = 0.0
            for q in range(start[t - lo], start[t - lo + 1]):
                # absolute position: q itself when the buffer starts at
                # token 0, otherwise ≥ 4 letters of history are present
                # wildcard letter: constant charge (0); letters after a
                # wildcard back off to the context that starts past it
                if wbuf[q]:
                    continue
                k = 5
                if lo == 0 and q + 1 < 5:
                    k = q + 1
                for r in range(q - 1, q - k, -1):
                    if wbuf[r]:
                        k = q - r
                        break
                code = 0
                for r in range(q - k + 1, q):
                    code = code * A + buf[r]
                tot += tabs[tab_off[k] + code * A + buf[q]]
            d_ll += tot - tokll[t]
            out_tok[n_out] = t
            out_val[n_out] = tot
            n_out += 1
        # repeat-rule pairs charged to this occurrence: (p-1, p) and,
        # unless the next occurrence is p+1, (p, p+1)
        for side in range(2):
            a = p - 1 + side
            if a < 0 or a + 1 >= n_tok:
                continue
            if side == 1 and nxt == p + 1:
                continue
            if not adj[a]:
                continue
            sa = symbols[a]
            sb = symbols[a + 1]
            if sa == sb:
                continue
            ua = units[a]
            ub = units[a + 1]
            na = ua
            nb = ub
            for c in range(n_ch):
                if ch_sym[c] == sa:
                    na = ch_unit[c]
                if ch_sym[c] == sb:
                    nb = ch_unit[c]
            d_viol += (1 if na == nb else 0) - (1 if ua == ub else 0)
    return d_ll, d_viol, n_out


@njit(cache=True, nogil=True)
def _xlogx_nb(c):
    return c * math.log(c) if c > 0 else 0.0


@njit(cache=True, nogil=True)
def _apply_counts(
    ch_sym, ch_unit, n_ch, sym_map, occ, first, second, cnt, log_prior, S1, S2, n, sign
):
    """Apply (sign=+1) or revert (sign=-1) the letter-count change of the
    reassignments to ``cnt`` in place; returns the updated (S1, S2, n)."""
    for c in range(n_ch):
        s = ch_sym[c]
        k = occ[s] * sign
        o = sym_map[s]
        u = ch_unit[c]
        for pair in range(2):
            unit = o if pair == 0 else u
            d = -k if pair == 0 else k
            for slot in range(2):
                a = first[unit] if slot == 0 else second[unit]
                if a < 0:
                    continue
                cur = cnt[a]
                S1 += _xlogx_nb(cur + d) - _xlogx_nb(cur)
                S2 += d * log_prior[a]
                n += d
                cnt[a] = cur + d
    return S1, S2, n


@njit(cache=True, nogil=True)
def _merge_sorted(a, b, out):
    i = 0
    j = 0
    k = 0
    while i < a.shape[0] and j < b.shape[0]:
        if a[i] < b[j]:
            out[k] = a[i]
            i += 1
        else:
            out[k] = b[j]
            j += 1
        k += 1
    while i < a.shape[0]:
        out[k] = a[i]
        i += 1
        k += 1
    while j < b.shape[0]:
        out[k] = b[j]
        j += 1
        k += 1
    return k


@njit(cache=True, nogil=True)
def _sa_run(
    sym_map,
    symbols,
    units,
    occ_idx,
    occ_ptr,
    occ,
    first,
    second,
    tabs,
    tab_off,
    tokll,
    adj,
    wild,
    wild_sym,
    cnt,
    log_prior,
    state,
    w_kl,
    w_rep,
    cdf,
    steps,
    t_start,
    t_end,
    seed,
):
    """The SA loop of ``WordHomophonicHead.sa_phase`` in one compiled pass.
    ``state = [ll, S1, S2, n, viol]`` is updated in place; returns
    ``(best_map, best, n_evals)``. Proposals: 90 % single reassignment of an
    occurrence-weighted symbol, 10 % swap of two symbols' units."""
    np.random.seed(seed)
    n_tok = symbols.shape[0]
    n_units = first.shape[0]
    out_tok = np.empty(2 * n_tok + 8, np.int64)
    out_val = np.empty(2 * n_tok + 8, np.float64)
    pos_buf = np.empty(n_tok + 8, np.int64)
    ch_sym = np.empty(2, np.int64)
    ch_unit = np.empty(2, np.int64)
    ll, S1, S2, viol = state[0], state[1], state[2], state[4]
    n = int(state[3])
    score = ll - w_kl * (S1 - n * math.log(n) - S2) - w_rep * viol
    best = score
    best_map = sym_map.copy()
    log_ratio = math.log(t_end / t_start) / max(steps - 1, 1)
    n_evals = 0
    for step in range(steps):
        t = t_start * math.exp(step * log_ratio)
        if np.random.random() < 0.9:
            s = np.searchsorted(cdf, np.random.random())
            new = np.random.randint(n_units)
            if new == sym_map[s] or wild_sym[s]:
                continue
            n_ch = 1
            ch_sym[0] = s
            ch_unit[0] = new
            pos = occ_idx[occ_ptr[s] : occ_ptr[s + 1]]
        else:
            s1 = np.searchsorted(cdf, np.random.random())
            s2 = np.searchsorted(cdf, np.random.random())
            if sym_map[s1] == sym_map[s2] or wild_sym[s1] or wild_sym[s2]:
                continue
            n_ch = 2
            ch_sym[0] = s1
            ch_unit[0] = sym_map[s2]
            ch_sym[1] = s2
            ch_unit[1] = sym_map[s1]
            m = _merge_sorted(
                occ_idx[occ_ptr[s1] : occ_ptr[s1 + 1]],
                occ_idx[occ_ptr[s2] : occ_ptr[s2 + 1]],
                pos_buf,
            )
            pos = pos_buf[:m]
        d_ll, dv, n_out = _scan(
            pos,
            pos.shape[0],
            ch_sym[:n_ch],
            ch_unit[:n_ch],
            symbols,
            units,
            first,
            second,
            tabs,
            tab_off,
            tokll,
            n_tok,
            adj,
            wild,
            out_tok,
            out_val,
        )
        S1n, S2n, nn = _apply_counts(
            ch_sym,
            ch_unit,
            n_ch,
            sym_map,
            occ,
            first,
            second,
            cnt,
            log_prior,
            S1,
            S2,
            n,
            1,
        )
        new_score = (
            (ll + d_ll) - w_kl * (S1n - nn * math.log(nn) - S2n) - w_rep * (viol + dv)
        )
        d = new_score - score
        n_evals += 1
        if d > 0 or np.random.random() < math.exp(d / t):
            for c in range(n_ch):
                sym_map[ch_sym[c]] = ch_unit[c]
            for i in range(pos.shape[0]):
                units[pos[i]] = sym_map[symbols[pos[i]]]
            for i in range(n_out):
                tokll[out_tok[i]] = out_val[i]
            S1, S2, n = S1n, S2n, nn
            viol += dv
            ll += d_ll
            score = new_score
            if score > best:
                best = score
                best_map[:] = sym_map
        else:
            # revert the count change (sym_map still holds the old units)
            S1, S2, n = _apply_counts(
                ch_sym,
                ch_unit,
                n_ch,
                sym_map,
                occ,
                first,
                second,
                cnt,
                log_prior,
                S1n,
                S2n,
                nn,
                -1,
            )
    state[0], state[1], state[2], state[3], state[4] = ll, S1, S2, n, viol
    return best_map, best, n_evals


@njit(cache=True, nogil=True)
def _polish_run(
    sym_map,
    symbols,
    units,
    occ_idx,
    occ_ptr,
    occ,
    first,
    second,
    tabs,
    tab_off,
    tokll,
    adj,
    wild,
    wild_sym,
    cnt,
    log_prior,
    state,
    w_kl,
    w_rep,
    max_sweeps,
):
    """Greedy best-improvement sweeps (``WordHomophonicHead.polish``) in
    one compiled pass; returns ``(score, n_evals)``."""
    n_tok = symbols.shape[0]
    n_sym = sym_map.shape[0]
    n_units = first.shape[0]
    out_tok = np.empty(2 * n_tok + 8, np.int64)
    out_val = np.empty(2 * n_tok + 8, np.float64)
    ch_sym = np.empty(1, np.int64)
    ch_unit = np.empty(1, np.int64)
    ll, S1, S2, viol = state[0], state[1], state[2], state[4]
    n = int(state[3])
    score = ll - w_kl * (S1 - n * math.log(n) - S2) - w_rep * viol
    n_evals = 0
    for _ in range(max_sweeps):
        improved = False
        for s in range(n_sym):
            if wild_sym[s]:
                continue
            cur = sym_map[s]
            pos = occ_idx[occ_ptr[s] : occ_ptr[s + 1]]
            ch_sym[0] = s
            best_u = -1
            best_sc = score + 1e-9
            for u in range(n_units):
                if u == cur:
                    continue
                ch_unit[0] = u
                d_ll, dv, n_out = _scan(
                    pos,
                    pos.shape[0],
                    ch_sym,
                    ch_unit,
                    symbols,
                    units,
                    first,
                    second,
                    tabs,
                    tab_off,
                    tokll,
                    n_tok,
                    adj,
                    wild,
                    out_tok,
                    out_val,
                )
                S1n, S2n, nn = _apply_counts(
                    ch_sym,
                    ch_unit,
                    1,
                    sym_map,
                    occ,
                    first,
                    second,
                    cnt,
                    log_prior,
                    S1,
                    S2,
                    n,
                    1,
                )
                sc = (
                    (ll + d_ll)
                    - w_kl * (S1n - nn * math.log(nn) - S2n)
                    - w_rep * (viol + dv)
                )
                S1, S2, n = _apply_counts(
                    ch_sym,
                    ch_unit,
                    1,
                    sym_map,
                    occ,
                    first,
                    second,
                    cnt,
                    log_prior,
                    S1n,
                    S2n,
                    nn,
                    -1,
                )
                n_evals += 1
                if sc > best_sc:
                    best_sc = sc
                    best_u = u
            if best_u >= 0:
                ch_unit[0] = best_u
                d_ll, dv, n_out = _scan(
                    pos,
                    pos.shape[0],
                    ch_sym,
                    ch_unit,
                    symbols,
                    units,
                    first,
                    second,
                    tabs,
                    tab_off,
                    tokll,
                    n_tok,
                    adj,
                    wild,
                    out_tok,
                    out_val,
                )
                S1, S2, n = _apply_counts(
                    ch_sym,
                    ch_unit,
                    1,
                    sym_map,
                    occ,
                    first,
                    second,
                    cnt,
                    log_prior,
                    S1,
                    S2,
                    n,
                    1,
                )
                sym_map[s] = best_u
                for i in range(pos.shape[0]):
                    units[pos[i]] = best_u
                for i in range(n_out):
                    tokll[out_tok[i]] = out_val[i]
                viol += dv
                ll += d_ll
                score = ll - w_kl * (S1 - n * math.log(n) - S2) - w_rep * viol
                improved = True
        if not improved:
            break
    state[0], state[1], state[2], state[3], state[4] = ll, S1, S2, n, viol
    return score, n_evals


def _xlogx(c: float) -> float:
    return c * math.log(c) if c > 0 else 0.0


class WordHomObjectiveState:
    """See the module docstring. ``sym_map`` is the live key (mutated by
    :meth:`commit`); ``score`` the current penalized objective."""

    def __init__(self, head, symbols, adj, sym_map, language, targets):
        from .wordhom import expand_units, rule_violations

        if head.rescore_order != 5:
            raise NotImplementedError("incremental objective is pentagram-only")
        self.head = head
        self.w_kl = float(head.freq_penalty_weight)
        self.w_rep = float(head.repeat_weight)
        self.symbols = np.asarray(symbols, dtype=np.int64)
        self.adj = np.asarray(adj, dtype=bool)
        if len(self.adj) < len(self.symbols):  # pad so adj[n_tok-1] is addressable
            self.adj = np.concatenate([self.adj, np.zeros(1, dtype=bool)])
        self.sym_map = np.asarray(sym_map, dtype=np.int64).copy()
        wt = getattr(head, "wild_types", None)
        self.wild_sym = (
            np.zeros(len(self.sym_map), dtype=np.int64)
            if wt is None
            else np.asarray(wt, dtype=bool).astype(np.int64)
        )
        self.wild = self.wild_sym[self.symbols]
        self.language = language
        self.targets = targets
        self.first = targets.first
        self.second = targets.second
        self.first_l = self.first.tolist()
        self.second_l = self.second.tolist()
        self.n_units = targets.n
        n_sym = len(self.sym_map)
        self.n_tok = len(self.symbols)
        self.occ = np.bincount(self.symbols, minlength=n_sym)
        order = np.argsort(self.symbols, kind="stable")
        self.occ_idx = order  # positions grouped by symbol, ascending within
        self.occ_ptr = np.concatenate([[0], np.cumsum(self.occ)])
        lm = head.ev.lms[language]
        tabs = [lm.table(k).reshape(-1).astype(np.float64) for k in range(1, 6)]
        self.tab_off = np.zeros(6, dtype=np.int64)
        self.tab_off[1:] = np.cumsum([0] + [len(t) for t in tabs[:-1]])
        self.tabs = np.concatenate(tabs)
        self.log_prior = head._log_prior(language)
        self.log_prior_l = self.log_prior.tolist()
        self.units = self.sym_map[self.symbols]
        letters = expand_units(self.units, targets)
        self.cnt = np.bincount(letters, minlength=A).astype(np.int64)
        self.n = len(letters)
        self.S1 = float(sum(_xlogx(float(c)) for c in self.cnt))
        self.S2 = float((self.cnt * self.log_prior).sum())
        self.viol = rule_violations(
            self.units, self.symbols, self.adj[: self.n_tok - 1]
        )
        self._out_tok = np.empty(max(2 * self.n_tok, 8), dtype=np.int64)
        self._out_val = np.empty(max(2 * self.n_tok, 8), dtype=np.float64)
        self._empty_ch = np.zeros(0, dtype=np.int64)
        self.tokll = np.zeros(self.n_tok)
        _, _, n_out = self._run(
            np.arange(self.n_tok, dtype=np.int64), self._empty_ch, self._empty_ch
        )
        self.tokll[self._out_tok[:n_out]] = self._out_val[:n_out]
        self.ll = float(self.tokll.sum())
        self.score = self._total(self.ll, self.S1, self.S2, self.n, self.viol)
        self._pending = None

    # -- pieces --------------------------------------------------------------

    def _total(self, ll, S1, S2, n, viol):
        nkl = S1 - n * math.log(n) - S2 if n > 0 else 0.0
        return ll - self.w_kl * nkl - self.w_rep * viol

    def _run(self, pos, ch_sym, ch_unit):
        return _scan(
            pos,
            len(pos),
            ch_sym,
            ch_unit,
            self.symbols,
            self.units,
            self.first,
            self.second,
            self.tabs,
            self.tab_off,
            self.tokll,
            self.n_tok,
            self.adj,
            self.wild,
            self._out_tok,
            self._out_val,
        )

    def _positions(self, syms):
        if len(syms) == 1:
            s = syms[0]
            return self.occ_idx[self.occ_ptr[s] : self.occ_ptr[s + 1]]
        return np.sort(
            np.concatenate(
                [self.occ_idx[self.occ_ptr[s] : self.occ_ptr[s + 1]] for s in syms]
            )
        )

    def _count_terms(self, changes):
        """(dl dict letter→Δcount, S1', S2', n') for the changes."""
        dl: dict[int, int] = {}
        for s, u in changes:
            o = int(self.sym_map[s])
            k = int(self.occ[s])
            for unit, sign in ((o, -k), (u, k)):
                a = self.first_l[unit]
                dl[a] = dl.get(a, 0) + sign
                b = self.second_l[unit]
                if b >= 0:
                    dl[b] = dl.get(b, 0) + sign
        S1, S2, n = self.S1, self.S2, self.n
        cnt = self.cnt
        for a, d in dl.items():
            if d:
                c = int(cnt[a])
                S1 += _xlogx(c + d) - _xlogx(c)
                S2 += d * self.log_prior_l[a]
                n += d
        return dl, S1, S2, n

    # -- API -----------------------------------------------------------------

    def delta(self, syms, new_units) -> float:
        """Objective change from setting ``sym_map[syms] = new_units``
        (does not mutate; the evaluation is kept for :meth:`commit`)."""
        changes = [(int(s), int(u)) for s, u in zip(syms, new_units)]
        ch_sym = np.array([s for s, _ in changes], dtype=np.int64)
        ch_unit = np.array([u for _, u in changes], dtype=np.int64)
        pos = self._positions(ch_sym)
        d_ll, dv, n_out = self._run(pos, ch_sym, ch_unit)
        dl, S1, S2, n = self._count_terms(changes)
        viol = self.viol + dv
        new_score = self._total(self.ll + d_ll, S1, S2, n, viol)
        self._pending = (
            changes,
            pos,
            self._out_tok[:n_out].copy(),
            self._out_val[:n_out].copy(),
            self.ll + d_ll,
            dl,
            S1,
            S2,
            n,
            viol,
            new_score,
        )
        return new_score - self.score

    def commit(self, syms, new_units) -> float:
        """Apply the move (re-evaluating it unless it is the one
        :meth:`delta` just scored) and return the new score."""
        changes = [(int(s), int(u)) for s, u in zip(syms, new_units)]
        if self._pending is None or self._pending[0] != changes:
            self.delta(syms, new_units)
        _, pos, toks, vals, ll, dl, S1, S2, n, viol, new_score = self._pending
        for s, u in changes:
            self.sym_map[s] = u
        if len(pos):
            self.units[pos] = self.sym_map[self.symbols[pos]]
        self.tokll[toks] = vals
        for a, d in dl.items():
            self.cnt[a] += d
        self.ll, self.S1, self.S2, self.n, self.viol = ll, S1, S2, n, viol
        self.score = new_score
        self._pending = None
        return new_score

    # -- compiled loops -------------------------------------------------------

    def _state_vec(self):
        return np.array([self.ll, self.S1, self.S2, float(self.n), float(self.viol)])

    def _sync(self, state):
        self.ll, self.S1, self.S2 = float(state[0]), float(state[1]), float(state[2])
        self.n, self.viol = int(state[3]), round(state[4])
        self.score = self._total(self.ll, self.S1, self.S2, self.n, self.viol)
        self._pending = None

    def sa(self, rng, steps, t_start, t_end, cdf):
        """Run the SA loop in compiled code on the live state; returns
        ``(best_map, best_score, n_evals)``. The state ends at the LAST
        accepted key (not the best)."""
        state = self._state_vec()
        best_map, best, n_evals = _sa_run(
            self.sym_map,
            self.symbols,
            self.units,
            self.occ_idx,
            self.occ_ptr,
            self.occ,
            self.first,
            self.second,
            self.tabs,
            self.tab_off,
            self.tokll,
            self.adj,
            self.wild,
            self.wild_sym,
            self.cnt,
            self.log_prior,
            state,
            self.w_kl,
            self.w_rep,
            np.asarray(cdf, dtype=np.float64),
            int(steps),
            float(t_start),
            float(t_end),
            int(rng.integers(2**31 - 1)),
        )
        self._sync(state)
        return best_map, float(best), int(n_evals)

    def polish(self, max_sweeps):
        """Greedy best-improvement sweeps in compiled code on the live
        state; returns ``(score, n_evals)``."""
        state = self._state_vec()
        score, n_evals = _polish_run(
            self.sym_map,
            self.symbols,
            self.units,
            self.occ_idx,
            self.occ_ptr,
            self.occ,
            self.first,
            self.second,
            self.tabs,
            self.tab_off,
            self.tokll,
            self.adj,
            self.wild,
            self.wild_sym,
            self.cnt,
            self.log_prior,
            state,
            self.w_kl,
            self.w_rep,
            int(max_sweeps),
        )
        self._sync(state)
        return float(score), int(n_evals)

    def deltas_all(self, s: int) -> np.ndarray:
        """(n_units,) objective change for every reassignment of symbol
        ``s`` (0 at its current unit)."""
        s = int(s)
        cur = int(self.sym_map[s])
        C = self.n_units
        pos = self.occ_idx[self.occ_ptr[s] : self.occ_ptr[s + 1]]
        ch_sym = np.array([s], dtype=np.int64)
        d_ll = np.zeros(C)
        dv = np.zeros(C)
        for c in range(C):
            if c == cur:
                continue
            d_ll[c], dv[c], _ = self._run(pos, ch_sym, np.array([c], dtype=np.int64))
        # letter counts per candidate
        k = int(self.occ[s])
        cnt = np.repeat(self.cnt[None, :], C, 0)
        fo, so = self.first[cur], self.second[cur]
        cnt[:, fo] -= k
        if so >= 0:
            cnt[:, so] -= k
        ar = np.arange(C)
        cnt[ar, self.first] += k
        sb = self.second >= 0
        cnt[np.flatnonzero(sb), self.second[sb]] += k
        n = cnt.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            S1 = np.where(cnt > 0, cnt * np.log(cnt), 0.0).sum(1)
        S2 = cnt @ self.log_prior
        nkl = S1 - n * np.log(n) - S2
        new_score = self.ll + d_ll - self.w_kl * nkl - self.w_rep * (self.viol + dv)
        d = new_score - self.score
        d[cur] = 0.0
        return d
