"""Alternating n-gram ↔ diffusion key search (``docs/alt_loop_plan.md``).

One round: a *proposer* rewrites part of the key (the diffusion posterior's
disagreement set, a random re-seed of the same size, or a racing ELBO
polish), then the n-gram search runs a short low-temperature SA + polish
from the rewritten key, and the round is accepted iff the **n-gram**
penalized objective improved. The judge proposes, the n-gram disposes;
ground truth is recorded per round but never consulted by the loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .posterior import disagreements

MECHANISMS = (
    "posterior",
    "posterior_sample",
    "random",
    "race",
    "none",
    "pair_swap",
    "random_swap",
)


def pair_swaps(
    scores: np.ndarray, key: np.ndarray, k: int
) -> list[tuple[int, int, float]]:
    """Top-``k`` disjoint key-index transpositions ranked by the judge's
    summed posterior gain — the bijection-preserving proposer for the 1:1
    head (``docs/altloop_vms_plan.md`` §2). ``key[i]`` is the letter of
    index ``i``; indices with no occurrences (a 25-slot injective key on a
    20-symbol stream) have all-zero score rows, so a swap with one of them
    is scored as the plain reassignment of the occupied index. Returns
    ``(i, j, gain)`` with ``gain > 0`` only."""
    key = np.asarray(key, dtype=np.int64)
    n = len(key)
    own = scores[np.arange(n), key]  # support for the current letters
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            g = scores[i, key[j]] + scores[j, key[i]] - own[i] - own[j]
            if np.isfinite(g) and g > 0:
                cand.append((i, j, float(g)))
    cand.sort(key=lambda x: -x[2])
    used, out = set(), []
    for i, j, g in cand:
        if i in used or j in used:
            continue
        used.update((i, j))
        out.append((i, j, g))
        if len(out) >= k:
            break
    return out


# -- "promising" tiers (docs/altloop_vms_plan.md §5, fixed before any
# manuscript number was read) ------------------------------------------------

TIERS = ("PENDING", "NOISE", "NOTABLE", "PROMISING", "LANGUAGE-LIKE")
REF_VMS_CEILING = 1.25  # highest Phase-6 manuscript structure margin (1.249)
REF_TRUE_MIN = 1.49  # lowest true decipherment in the Phase-6 battery
NOTABLE_MIN = 1.26
NOTABLE_ABOVE_CONTROLS = 0.15
ABSTAIN_MAX_PLAIN = 3.0
ABSTAIN_MIN_MARGIN = 1.5


def classify_tier(
    margin: float,
    plain_bits: float,
    controls_best: float | None,
    *,
    flip_rate: float | None = None,
    controls_language_like: bool = False,
) -> str:
    """§5: the tier of one ``psamp`` reading given the *best* structure
    margin either control arm reached on the same cell. ``controls_best``
    None = a control has not reported yet → PENDING (never flag on a
    missing control). Anything a control also reaches is NOISE whatever
    the number. ``flip_rate`` None = replicate seeds not yet in — the
    LANGUAGE-LIKE tier needs it to be 0."""
    if controls_best is None:
        return "PENDING"
    if margin < NOTABLE_MIN or margin < controls_best + NOTABLE_ABOVE_CONTROLS:
        return "NOISE"
    tier = "NOTABLE"
    if margin >= REF_TRUE_MIN and controls_best < NOTABLE_MIN:
        tier = "PROMISING"
        if (
            plain_bits <= ABSTAIN_MAX_PLAIN
            and margin >= ABSTAIN_MIN_MARGIN
            and flip_rate is not None
            and flip_rate == 0
            and not controls_language_like
        ):
            tier = "LANGUAGE-LIKE"
    return tier


def alternate(
    key: np.ndarray,
    *,
    mechanism: str,
    objective: Callable[[np.ndarray], float],
    short_sa: Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, float]],
    scores_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    race_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    random_unit: Callable[[int, int, np.random.Generator], int] | None = None,
    metrics: Callable[[np.ndarray], dict] | None = None,
    occ: np.ndarray | None = None,
    temperature: float = 1.0,
    k: int | None = 8,
    rounds: int = 6,
    patience: int = 2,
    seed: int = 0,
    on_round: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Returns the best key by n-gram objective and a per-round trace.

    ``scores_fn(key) -> (n_sym, n_units)`` judge support (mechanism
    ``posterior``); ``k`` symbols re-seeded per round (``None`` = the whole
    disagreement set); ``random_unit(sym, cur_unit, rng)`` draws a unit for
    the ``random`` control (same length class as ``cur_unit``); ``race_fn``
    is the ELBO polish for mechanism ``race``; ``none`` runs the short SA
    alone (the "is it just the extra SA?" control). ``pair_swap`` /
    ``random_swap`` are the bijection-preserving forms (``k`` transpositions
    of key indices, judge-ranked / uniformly random). ``on_round(info)`` is
    called after every completed round (streaming metrics).
    """
    if mechanism not in MECHANISMS:
        raise ValueError(mechanism)
    rng = np.random.default_rng(seed)
    cur = np.asarray(key, dtype=np.int64).copy()
    cur_obj = float(objective(cur))
    trace = []
    start_metrics = metrics(cur) if metrics else {}
    stale = 0
    for r in range(rounds):
        t0 = time.time()
        prop = cur.copy()
        info = {"round": r, "obj_in": cur_obj}
        if mechanism == "posterior":
            D = disagreements(scores_fn(cur), cur)
            info["n_disagree"] = len(D)
            take = D if k is None else D[:k]
            for s, u, _ in take:
                prop[s] = u
            info["reseeded"] = [(s, int(cur[s]), u, round(m, 3)) for s, u, m in take]
            if not take:
                info["seconds"] = time.time() - t0
                trace.append(info)
                break
        elif mechanism == "posterior_sample":
            # judge picks the symbols (disagreement ranking); the unit is
            # SAMPLED from the per-occurrence mean posterior so repeated
            # rounds explore instead of re-proposing the same argmax
            S = scores_fn(cur)
            D = disagreements(S, cur)
            info["n_disagree"] = len(D)
            take = D if k is None else D[:k]
            info["reseeded"] = []
            for s, _, m in take:
                row = S[s] / max(float(occ[s]) if occ is not None else 1.0, 1.0)
                row = row / temperature
                fin = np.isfinite(row)
                pr = np.zeros_like(row)
                pr[fin] = np.exp(row[fin] - row[fin].max())
                pr[cur[s]] = 0.0
                if pr.sum() <= 0:
                    continue
                u = int(rng.choice(len(pr), p=pr / pr.sum()))
                prop[s] = u
                info["reseeded"].append((s, int(cur[s]), u, round(m, 3)))
            if not take:
                info["seconds"] = time.time() - t0
                trace.append(info)
                break
        elif mechanism == "random":
            n_take = k if k is not None else 8
            syms = rng.choice(len(cur), size=min(n_take, len(cur)), replace=False)
            info["reseeded"] = []
            for s in syms:
                u = random_unit(int(s), int(cur[s]), rng)
                info["reseeded"].append((int(s), int(cur[s]), int(u), None))
                prop[s] = u
        elif mechanism == "pair_swap":
            n_take = k if k is not None else 4
            sw = pair_swaps(scores_fn(cur), cur, n_take)
            info["reseeded"] = []
            for i, j, g in sw:
                prop[i], prop[j] = cur[j], cur[i]
                info["reseeded"].append((int(i), int(cur[i]), int(cur[j]), round(g, 3)))
            if not sw:
                info["seconds"] = time.time() - t0
                trace.append(info)
                if on_round:
                    on_round(info)
                break
        elif mechanism == "random_swap":
            n_take = k if k is not None else 4
            idx = rng.permutation(len(cur))[: 2 * n_take]
            info["reseeded"] = []
            for i, j in zip(idx[0::2], idx[1::2]):
                prop[i], prop[j] = cur[j], cur[i]
                info["reseeded"].append((int(i), int(cur[i]), int(cur[j]), None))
        elif mechanism == "race":
            prop = np.asarray(race_fn(cur), dtype=np.int64)
            info["reseeded"] = [
                (int(s), int(cur[s]), int(prop[s]), None)
                for s in np.flatnonzero(prop != cur)
            ]
        info["obj_proposed"] = float(objective(prop))
        info["n_changed_by_proposer"] = int((prop != cur).sum())
        new, new_obj = short_sa(prop, rng)
        new = np.asarray(new, dtype=np.int64)
        info["obj_after_sa"] = float(new_obj)
        info["n_changed_by_sa"] = int((new != prop).sum())
        info["n_changed_total"] = int((new != cur).sum())
        accepted = new_obj > cur_obj + 1e-9
        info["accepted"] = bool(accepted)
        if metrics:
            info["metrics_proposed"] = metrics(prop)
            info["metrics_after_sa"] = metrics(new)
        if accepted:
            cur, cur_obj, stale = new, float(new_obj), 0
        else:
            stale += 1
        info["obj_out"] = cur_obj
        info["seconds"] = time.time() - t0
        trace.append(info)
        if on_round:
            on_round(info)
        if stale >= patience:
            break
    return cur, {
        "mechanism": mechanism,
        "k": k,
        "start_obj": float(objective(key)),
        "final_obj": cur_obj,
        "start_metrics": start_metrics,
        "final_metrics": metrics(cur) if metrics else {},
        "n_rounds": len(trace),
        "n_accepted": sum(t.get("accepted", False) for t in trace),
        "trace": trace,
    }
