"""Shared plumbing for the Phase-5 ladder scripts (tasks 5.2–5.6).

- forked single-thread CPU worker pools for the inner n-gram searches
  (the CH-track discipline: ``torch.set_num_threads(1)``, nice(10));
- atomic JSON artifacts with resume-by-key;
- Wilson intervals; instance keys;
- the soft-refinement primitives of the outer tier (R3): gradient steps on a
  head's soft parameterization through the frozen diffusion evaluator,
  starting from a hard key. Rung 1 (Sinkhorn / permutation) and rung 2
  (row-stochastic symbol map) share :func:`refine_assignment`; rung 3's
  block refinement lives with its head.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .frame import letters_to_vocab, straight_through_frame
from .ngram import A
from .rung1_sinkhorn import sinkhorn


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def load_done(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {tuple(r[k] for k in key_fields): r for r in d.get("instances", [])}


def _init_worker():
    os.nice(10)
    torch.set_num_threads(1)


def run_pool(fn, jobs: list, *, workers: int, on_result=None, chunksize: int = 1):
    """Fork-pool map with incremental callback (results in completion order)."""
    torch.set_num_threads(1)
    ctx = mp.get_context("fork")
    out = []
    t0 = time.time()
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if on_result is not None:
                on_result(i, r, time.time() - t0)
    return out


# -- outer-tier soft refinement (R3) ----------------------------------------


def refine_assignment(
    evaluator,
    cipher_ids: np.ndarray,
    sym_to_letter: np.ndarray,
    *,
    language: str,
    bijective: bool,
    steps: int = 20,
    lr: float = 0.1,
    n_strata: int = 4,
    init_scale: float = 4.0,
    tau: float = 1.0,
    seed: int = 0,
    freq_kl_weight: float = 0.0,
    prior_log_unigram: np.ndarray | None = None,
    straight_through: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Gradient refinement of a hard symbol→letter map through the frozen
    diffusion evaluator (expected-embedding inputs, design §8).

    Parameterization: logits (n_symbols, A) initialised at ``init_scale`` ×
    one-hot of the start map; rows are Sinkhorn-normalised (bijective, rung
    1) or softmax-normalised (homophonic, rung 2). Each step scores the soft
    decode with a fresh masking seed (stochastic gradient over masks). The
    returned map is the Hungarian / argmax projection of the final logits;
    the trajectory of soft losses is returned for the record.
    """
    cipher_t = torch.from_numpy(np.asarray(cipher_ids, dtype=np.int64))
    n_sym = int(sym_to_letter.shape[0])
    L = len(cipher_t)
    logits = torch.full((n_sym, A), 0.0)
    logits[torch.arange(n_sym), torch.from_numpy(sym_to_letter.astype(np.int64))] = (
        init_scale
    )
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    losses = []
    prior = None
    if freq_kl_weight > 0 and prior_log_unigram is not None:
        prior = torch.from_numpy(np.asarray(prior_log_unigram, dtype=np.float32))
        occ = torch.bincount(cipher_t, minlength=n_sym).float()
    for step in range(steps):
        p = sinkhorn(logits / tau) if bijective else torch.softmax(logits / tau, dim=1)
        soft = p[cipher_t]
        frame = letters_to_vocab(soft)
        if straight_through:  # design §8 fallback: hard forward, soft backward
            frame = straight_through_frame(frame)
        score = evaluator.score_frame(
            frame,
            language=language,
            seed=seed + step,
            n_strata=n_strata,
        )
        loss = -score / L
        if prior is not None:
            emitted = (p * occ[:, None]).sum(0) / occ.sum()
            kl = (emitted * (torch.log(emitted.clamp_min(1e-9)) - prior)).sum()
            loss = loss + freq_kl_weight * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        if bijective:
            cost = -sinkhorn(logits / tau).log().clamp_min(-30)
            rows, cols = linear_sum_assignment(cost.numpy())
            out = np.empty(n_sym, dtype=np.int64)
            out[rows] = cols
        else:
            out = logits.argmax(dim=1).numpy().astype(np.int64)
    return out, losses


def elbo_gradient_search(
    evaluator,
    cipher_ids: np.ndarray,
    *,
    language: str,
    steps: int = 150,
    lr: float = 0.5,
    tau_start: float = 1.0,
    tau_end: float = 0.15,
    gumbel_scale: float = 0.5,
    n_strata: int = 4,
    seed: int = 0,
    straight_through: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """The rung-1 Gumbel–Sinkhorn gradient phase driven by the frozen
    DIFFUSION evaluator instead of the n-gram (R3 probe: can the ELBO's
    dense gradients find a 1:1 key from scratch?). Returns the Hungarian
    projection of the final logits and the loss trajectory."""
    cipher_t = torch.from_numpy(np.asarray(cipher_ids, dtype=np.int64))
    L = len(cipher_t)
    g = torch.Generator().manual_seed(seed)
    logits = torch.zeros(A, A, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    losses = []
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        tau = tau_start * (tau_end / tau_start) ** frac
        noise = -torch.log(-torch.log(torch.rand(A, A, generator=g).clamp_min(1e-20)))
        p = sinkhorn((logits + gumbel_scale * (1 - frac) * noise) / tau)
        frame = letters_to_vocab(p[cipher_t])
        if straight_through:
            frame = straight_through_frame(frame)
        score = evaluator.score_frame(
            frame, language=language, seed=seed + step, n_strata=n_strata
        )
        loss = -score / L
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        cost = -sinkhorn(logits / tau_end).log().clamp_min(-30)
    rows, cols = linear_sum_assignment(cost.numpy())
    perm = np.empty(A, dtype=np.int64)
    perm[rows] = cols
    return perm, losses


def refine_assignment_tracked(
    evaluator,
    cipher_ids: np.ndarray,
    sym_to_letter: np.ndarray,
    *,
    language: str,
    bijective: bool,
    steps: int = 40,
    lr: float = 0.05,
    n_strata: int = 32,
    init_scale: float = 6.0,
    fixed_masks: bool = True,
    straight_through: bool = False,
    eval_every: int = 5,
    eval_strata: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Conditioned refinement: one FIXED masking realization (a deterministic
    surrogate objective, ``fixed_masks``) or fresh masks per step; the hard
    projection is scored every ``eval_every`` steps with paired masks against
    the start key and the best hard key seen is returned (never worse than
    the start on the evaluation masks)."""
    from .two_tier import paired_bits

    cipher_np = np.asarray(cipher_ids, dtype=np.int64)
    cipher_t = torch.from_numpy(cipher_np)
    n_sym = int(sym_to_letter.shape[0])
    L = len(cipher_t)
    logits = torch.zeros(n_sym, A)
    logits[torch.arange(n_sym), torch.from_numpy(sym_to_letter.astype(np.int64))] = (
        init_scale
    )
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)

    def project(lg):
        with torch.no_grad():
            if bijective:
                cost = -sinkhorn(lg).log().clamp_min(-30)
                rows, cols = linear_sum_assignment(cost.numpy())
                out = np.empty(n_sym, dtype=np.int64)
                out[rows] = cols
                return out
            return lg.argmax(dim=1).numpy().astype(np.int64)

    start = sym_to_letter.astype(np.int64)
    best_key, best_bits = start.copy(), None
    trace = []
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            key = project(logits.detach())
            b = paired_bits(
                evaluator,
                np.stack([key[cipher_np], start[cipher_np]]),
                [language],
                n_strata=eval_strata,
                seed=seed + 99,
            )[:, 0]
            trace.append(
                {
                    "step": step,
                    "bits": float(b[0]),
                    "bits_start": float(b[1]),
                    "n_changed": int((key != start).sum()),
                }
            )
            if best_bits is None or b[0] < best_bits:
                best_bits, best_key = float(b[0]), key
        if step == steps:
            break
        p = sinkhorn(logits) if bijective else torch.softmax(logits, dim=1)
        frame = letters_to_vocab(p[cipher_t])
        if straight_through:
            frame = straight_through_frame(frame)
        score = evaluator.score_frame(
            frame,
            language=language,
            seed=seed if fixed_masks else seed + step,
            n_strata=n_strata,
        )
        loss = -score / L
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best_key, {"trace": trace, "best_bits": best_bits}


CHOICE_TERM_WARNING = (
    "choice_fn was given without choice_term_in_polish=True. The homophonic "
    "choice-bits term is a CELL-RANKING device (it penalises verbose decodes); "
    "inside a fixed-length symbol-map polish it only rewards moving a frequent "
    "symbol onto a spare letter and it DESTROYED the Borg decipherment "
    "(SER 0.12 -> 0.31; docs/race_polish_plan.md section 7). Polish on the ELBO "
    "alone (choice_fn=None); pass choice_term_in_polish=True only to reproduce "
    "the recorded Phase-5/6 behaviour."
)


def _check_choice_term(choice_fn, choice_term_in_polish: bool):
    if choice_fn is not None and not choice_term_in_polish:
        raise ValueError(CHOICE_TERM_WARNING)


def elbo_polish(
    evaluator,
    cipher_ids: np.ndarray,
    sym_map: np.ndarray,
    *,
    language: str,
    choice_fn=None,
    sweeps: int = 8,
    budget: int = 8,
    confirm_budget: int = 64,
    batch: int = 96,
    seed: int = 0,
    pair_swaps: bool = True,
    resample_masks: bool = False,
    set_moves: bool = True,
    choice_term_in_polish: bool = False,
) -> tuple[np.ndarray, dict]:
    """Discrete outer-tier refinement of a symbol→letter map under the frozen
    diffusion evaluator: every single-symbol reassignment (and symbol-pair
    letter swap) of the current map is decoded and scored in one paired
    batch (same masks for all candidates), with ``choice_fn(map, decode)``
    adding the cipher's choice bits per plaintext char (the MDL total of
    ``scale.py``); the best move is taken, repeated up to ``sweeps`` times.
    The result is confirmed against the start map at ``confirm_budget``
    (paired) and the start map is returned if it is not better there.
    ``set_moves=False`` restricts the neighbourhood to pair swaps, which
    keeps a bijective (1:1) key bijective.
    This is where the ELBO can improve on the n-gram objective's own
    optimum (rung 2: the pentagram's best map is not always the true map
    at 408 chars; the ELBO prefers the true one).

    ``choice_fn`` is OFF by default and refused unless
    ``choice_term_in_polish=True``: see ``CHOICE_TERM_WARNING`` — the MDL
    choice term belongs to the cell ranking, not to the polish objective
    (finding of 2026-08-25, ``docs/race_polish_plan.md`` §7)."""
    from .two_tier import paired_bits

    _check_choice_term(choice_fn, choice_term_in_polish)
    cipher = np.asarray(cipher_ids, dtype=np.int64)
    n_sym = int(sym_map.shape[0])
    cur = np.asarray(sym_map, dtype=np.int64).copy()
    start = cur.copy()

    def mdl(maps, bits):
        return np.array(
            [
                b + (choice_fn(m, m[cipher]) if choice_fn else 0.0)
                for m, b in zip(maps, bits)
            ]
        )

    trace = []
    for sweep in range(sweeps):
        cands = [cur]
        moves = [None]
        for s in range(n_sym if set_moves else 0):
            for letter in range(A):
                if letter == cur[s]:
                    continue
                m = cur.copy()
                m[s] = letter
                cands.append(m)
                moves.append(("set", s, letter))
        if pair_swaps:
            for s1 in range(n_sym):
                for s2 in range(s1 + 1, n_sym):
                    if cur[s1] == cur[s2]:
                        continue
                    m = cur.copy()
                    m[s1], m[s2] = m[s2], m[s1]
                    cands.append(m)
                    moves.append(("swap", s1, s2))
        rows = np.stack([m[cipher] for m in cands])
        bits = paired_bits(
            evaluator,
            rows,
            [language],
            n_strata=budget,
            seed=seed + (101 * sweep if resample_masks else 0),
            batch=batch,
        )[:, 0]
        total = mdl(cands, bits)
        k = int(np.argmin(total))
        gain = float(total[0] - total[k])
        trace.append(
            {"sweep": sweep, "best_move": moves[k], "gain": gain, "n_cands": len(cands)}
        )
        if k == 0 or gain <= 0:
            break
        cur = cands[k]
    rows = np.stack([cur[cipher], start[cipher]])
    b = paired_bits(
        evaluator, rows, [language], n_strata=confirm_budget, seed=seed + 7, batch=batch
    )[:, 0]
    t = mdl([cur, start], b)
    info = {
        "trace": trace,
        "confirm_bits": [float(b[0]), float(b[1])],
        "confirm_mdl": [float(t[0]), float(t[1])],
        "accepted": bool(t[0] < t[1]),
        "n_changed": int((cur != start).sum()),
    }
    return (cur if t[0] < t[1] else start), info


# -- outer-tier discrete refinement, racing form ----------------------------


def _neighbourhood(cur: np.ndarray, *, set_moves: bool, pair_swaps: bool):
    """Row 0 is the current map; every single-symbol reassignment and
    symbol-pair letter swap follows (identical to ``elbo_polish``)."""
    n_sym = int(cur.shape[0])
    cands, moves = [cur], [None]
    for s in range(n_sym if set_moves else 0):
        for letter in range(A):
            if letter == cur[s]:
                continue
            m = cur.copy()
            m[s] = letter
            cands.append(m)
            moves.append(("set", s, letter))
    if pair_swaps:
        for s1 in range(n_sym):
            for s2 in range(s1 + 1, n_sym):
                if cur[s1] == cur[s2]:
                    continue
                m = cur.copy()
                m[s1], m[s2] = m[s2], m[s1]
                cands.append(m)
                moves.append(("swap", s1, s2))
    return cands, moves


def _z_bonferroni(alpha: float, n: int) -> float:
    from scipy.stats import norm

    return float(norm.isf(alpha / max(n, 1)))


def race_polish(
    evaluator,
    cipher_ids: np.ndarray,
    sym_map: np.ndarray,
    *,
    language: str,
    choice_fn=None,
    sweeps: int = 8,
    budgets: tuple[int, ...] = (4, 16, 64, 128),
    max_survivors: tuple[int | None, ...] = (None, 64, 12, 4),
    alpha: float = 0.05,
    se_floor_frac: float = 0.5,
    batch: int = 96,
    seed: int = 0,
    pair_swaps: bool = True,
    set_moves: bool = True,
    confirm_budget: int = 64,
    choice_term_in_polish: bool = False,
) -> tuple[np.ndarray, dict]:
    """Racing form of :func:`elbo_polish` (``docs/race_polish_plan.md``).

    The greedy polish takes the argmin over ~n_sym·A noisy paired estimates
    at one small budget and commits it — a winner's-curse selection that
    at Borg scale degraded the key every sweep. Here every neighbour is
    screened cheaply on the PAIRED DIFFERENCE to the current map (same
    masks, so the difference is low-variance and its spread over strata
    estimates the noise); candidates whose interval says "not an
    improvement" are eliminated with a Bonferroni-corrected z over the live
    set; survivors are re-scored with FRESH masks at growing budgets and the
    non-screen draws pooled; a move is committed only if it is confidently
    better than the current map on those fresh draws. If nothing qualifies
    the polish stops — "no real improvement here" is a first-class outcome.

    ``choice_fn`` is refused unless ``choice_term_in_polish=True`` (see
    ``CHOICE_TERM_WARNING``): with the term in the objective the race still
    degraded Borg (0.12 → 0.225); on the ELBO alone it held it (0.1194).
    """
    from .two_tier import paired_bits, paired_bits_strata

    _check_choice_term(choice_fn, choice_term_in_polish)
    cipher = np.asarray(cipher_ids, dtype=np.int64)
    cur = np.asarray(sym_map, dtype=np.int64).copy()
    start = cur.copy()
    n_stages = len(budgets)
    if len(max_survivors) != n_stages:
        raise ValueError("budgets and max_survivors must have the same length")

    def choice_delta(maps):
        if choice_fn is None:
            return np.zeros(len(maps))
        c0 = choice_fn(maps[0], maps[0][cipher])
        return np.array([choice_fn(m, m[cipher]) - c0 for m in maps])

    def stats(d):
        """d: [n_alive, K] per-stratum paired differences (row of the
        current map excluded). Mean and shrunk standard error."""
        K = d.shape[1]
        mean = d.mean(1)
        se = d.std(1, ddof=1) / np.sqrt(K) if K > 1 else np.full(len(d), np.inf)
        floor = se_floor_frac * float(np.median(se)) if len(se) else 0.0
        return mean, np.maximum(se, floor)

    trace = []
    draws_total = 0
    for sweep in range(sweeps):
        cands, moves = _neighbourhood(cur, set_moves=set_moves, pair_swaps=pair_swaps)
        n_cand = len(cands) - 1
        if n_cand == 0:
            break
        cdelta = choice_delta(cands)  # deterministic MDL choice term per candidate
        alive = np.arange(1, len(cands))
        pooled = None  # [n_alive, K_pooled] fresh (non-screen) draws
        stage_log = []
        committed = None
        for stage, (budget, cap) in enumerate(zip(budgets, max_survivors)):
            rows = np.stack([cands[i][cipher] for i in np.concatenate([[0], alive])])
            per = paired_bits_strata(
                evaluator,
                rows,
                [language],
                n_strata=budget,
                seed=seed + 1009 * stage + 7 * sweep,
                batch=batch,
            )[:, 0, :]
            draws_total += int(rows.shape[0] * budget)
            # per-stratum values are contributions (already / n_strata); scale
            # to per-stratum bits so pooled draws across budgets are commensurate
            d = (per[1:] - per[0:1]) * budget + cdelta[alive][:, None]
            if stage == 0:
                mean, se = stats(d)
            else:
                pooled = d if pooled is None else np.concatenate([pooled, d], 1)
                mean, se = stats(pooled)
            z = _z_bonferroni(alpha, len(alive))
            keep = (mean - z * se) <= 0  # cannot yet exclude "improvement"
            order = np.argsort(mean)
            capped = False
            if cap is not None and keep.sum() > cap:
                top = set(order[:cap].tolist())
                keep = np.array([keep[i] and i in top for i in range(len(alive))])
                capped = True
            stage_log.append(
                {
                    "stage": stage,
                    "budget": budget,
                    "n_alive_in": len(alive),
                    "n_alive_out": int(keep.sum()),
                    "z": z,
                    "capped": capped,
                    "best_mean": float(mean[order[0]]),
                    "best_se": float(se[order[0]]),
                }
            )
            # commit test on fresh draws only
            if stage > 0 and keep.any():
                zc = _z_bonferroni(alpha, int(keep.sum()))
                ok = np.where(keep & ((mean + zc * se) < 0))[0]
                if len(ok):
                    b = ok[np.argmin(mean[ok])]
                    committed = (int(alive[b]), float(mean[b]), float(se[b]), zc)
                    break
            if not keep.any():
                break
            idx = np.where(keep)[0]
            alive = alive[idx]
            if pooled is not None:
                pooled = pooled[idx]
        entry = {
            "sweep": sweep,
            "n_cands": n_cand,
            "stages": stage_log,
            "best_move": None,
            "gain": 0.0,
        }
        if committed is None:
            trace.append(entry)
            break
        i, m, s, zc = committed
        entry.update(
            best_move=moves[i],
            gain=float(-m),
            se=s,
            z=zc,
            changed=int((cands[i] != cur).sum()),
        )
        trace.append(entry)
        cur = cands[i]
    rows = np.stack([cur[cipher], start[cipher]])
    b = paired_bits(
        evaluator, rows, [language], n_strata=confirm_budget, seed=seed + 7, batch=batch
    )[:, 0]
    cd = choice_delta([start, cur])  # relative to start
    t = np.array([b[0] + cd[1], b[1] + cd[0]])
    draws_total += 2 * confirm_budget
    info = {
        "method": "race",
        "trace": trace,
        "confirm_bits": [float(b[0]), float(b[1])],
        "confirm_mdl": [float(t[0]), float(t[1])],
        "accepted": bool(t[0] < t[1]),
        "n_changed": int((cur != start).sum()),
        "n_moves": int(sum(1 for e in trace if e["best_move"] is not None)),
        "draws_total": int(draws_total),
    }
    return (cur if t[0] < t[1] else start), info
