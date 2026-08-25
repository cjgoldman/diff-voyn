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

MECHANISMS = ("posterior", "posterior_sample", "random", "race", "none")


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
) -> tuple[np.ndarray, dict]:
    """Returns the best key by n-gram objective and a per-round trace.

    ``scores_fn(key) -> (n_sym, n_units)`` judge support (mechanism
    ``posterior``); ``k`` symbols re-seeded per round (``None`` = the whole
    disagreement set); ``random_unit(sym, cur_unit, rng)`` draws a unit for
    the ``random`` control (same length class as ``cur_unit``); ``race_fn``
    is the ELBO polish for mechanism ``race``; ``none`` runs the short SA
    alone (the "is it just the extra SA?" control).
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
