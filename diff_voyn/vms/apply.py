"""Phase 6 — applying the validated heads to a ciphertext and ranking the
(cipher × language) cells (tasks 6.1–6.3; design §8 R5, §9; the Phase-5
carry-overs in ``docs/phase5_status.md``).

One pipeline for the manuscript and for every control instance:

solve (CPU pool)
    For each (instance, presentation, head, solve window, hypothesis): the
    inner n-gram search of the Phase-5 rung (rung 1 Sinkhorn + ILS; rung 2
    SA restarts + pair polish; rung 3 block Sinkhorn + fixed-parse polish;
    rung 4 segmented Sinkhorn + integer polish) returns a shortlist of
    distinct keys. Keys are solved on a WINDOW of the stream (the
    validated Phase-5 scales: 2–4k symbols, ~4k tokens) and applied to the
    whole dialect stream — a key is global per dialect, so solving on a
    window and decoding everything is ordinary cryptanalysis, and several
    windows per stream give a key-stability / agreement check.

score (GPU, frozen evaluator)
    Outer tier per job: every shortlist decode of the solve window scored
    under every language condition with paired masks (budget 64), selection
    by the MDL rule (calibrated plaintext bits + choice bits per plaintext
    char; key bits are constant within a job), ``elbo_polish`` for the
    homophonic head (the discrete outer tier that Phase 5 found works),
    then the chosen key decodes the FULL stream, which is scored on
    ≤ ``score_windows`` 1024-char windows × 4 replicate seeds under all
    conditions, each window paired with a letter-shuffled copy of itself
    (the Phase-3 per-instance structure control).

report
    Per cell: calibrated plaintext bits/char (mean ± s.e.m. over windows
    and seeds), key bits, choice bits, total per COVERED ciphertext symbol
    (the cross-head MDL comparator of §5.6) and per all ciphertext symbols
    with uncovered symbols charged at the stream's own order-0 entropy,
    coverage, structure margin (shuffled − decode, own condition), the
    replicate flip-rate of the language rank, the calibration margin
    uncertainty, and the abstention verdict. The abstention rule is fixed
    HERE, before any manuscript number is read (``ABSTAIN_RULE``): a cell
    is "language-like" only if its full-stream calibrated plaintext bits
    are at the clean-text level (≤ 3.0 bits/char — Phase-5 solved cells sit
    at 1.9–2.7, the wrong-key plateau at 3.4–4.4) AND its structure margin
    is in the true-decipherment band (≥ 1.5 bits/char — Phase 3 measured
    1.6–2.9 for true decipherments, 0.5–1.9 for wrong ones). A dialect
    whose every cell fails abstains; the language ranking is still
    reported, flagged as a ranking among non-decipherments.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ..data.loader import LANG_TO_INDEX
from ..heads.ladder import load_done, run_pool, write_json_atomic
from ..heads.ngram import A
from ..heads.scale import choice_bits, log2_factorial
from ..metrology import CalibrationTable, calibrate_bits

LANGS = tuple(LANG_TO_INDEX)
HEADS = ("sub1to1", "homophonic", "naibbe", "arithmetic")
# word-level homophonic head (Boxer's hypothesis without the arithmetic);
# runs on the ``wordtypes<K>`` presentations only, see heads/wordhom.py
WORDHOM = "wordhom"
KEY = ("instance", "presentation", "head", "window", "hypothesis")
ABSTAIN_RULE = {
    "max_plain_bits": 3.0,
    "min_structure_margin": 1.5,
    "note": "fixed before any manuscript number was read; see module docstring",
}
_EV = None
_PARSER = None


# -- jobs --------------------------------------------------------------------


def head_key_bits(head: str, n_sym: int) -> float:
    """Description length of the key class actually searched: an injective
    map of ``n_sym`` symbols into the 25 letters for rung 1 (log2 25!/(25−n)!),
    the Phase-5 terms otherwise."""
    if head == "sub1to1":
        return log2_factorial(A) - log2_factorial(max(A - n_sym, 0))
    if head == "homophonic":
        return n_sym * math.log2(A)
    if head == "naibbe":
        return 18 * log2_factorial(23)
    if head == "arithmetic":
        return log2_factorial(16) + A * math.log2(26)
    if head == WORDHOM:
        from ..heads.wordhom import N_BIGRAMS

        return n_sym * math.log2(A + N_BIGRAMS)
    raise ValueError(head)


def window_slices(n: int, length: int, n_windows: int) -> list[tuple[int, int]]:
    """``n_windows`` evenly spaced [a, b) solve windows over ``n`` units."""
    if n <= length or n_windows <= 1:
        return [(0, min(n, length))] if n > 0 else []
    starts = np.linspace(0, n - length, n_windows).astype(int)
    return [(int(s), int(s + length)) for s in starts]


def make_jobs(
    instance: dict,
    *,
    heads: tuple[str, ...],
    hypotheses: tuple[str, ...] = LANGS,
    n_windows: int = 2,
    w1: int = 4000,
    w2: int = 2000,
    w3: int = 4000,
    w4: int = 500,
    w5: int = 12000,
    restarts: dict | None = None,
    units: str | None = None,
) -> list[dict]:
    """Jobs for one instance (a presentation record: ``kind`` eva/boxer
    carries ``symbols`` (+ ``token_starts``), ``words`` carries ``tokens``).
    The stream itself is NOT copied into the job — workers read it from
    ``instance['path']``."""
    restarts = restarts or {}
    jobs = []
    kind = instance["kind"]
    for head in heads:
        if (head == WORDHOM) != kind.startswith("wordtypes"):
            continue
        if head == "naibbe" and kind != "words":
            continue
        if head == "arithmetic" and not (
            kind == "boxer" and instance["n_symbols"] == 16
        ):
            continue
        if head in ("sub1to1", "homophonic") and kind == "words":
            continue
        if head == "sub1to1" and instance["n_symbols"] > A:
            continue
        if head == "naibbe":
            n, length = instance["n_stream"], w3
        elif head == "arithmetic":
            n, length = instance["n_tokens_kept"], w4
        elif head == WORDHOM:
            n, length = instance["n_stream"], w5
        else:
            n, length = instance["n_stream"], w1 if head == "sub1to1" else w2
        for wi, (a, b) in enumerate(window_slices(n, length, n_windows)):
            for hyp in hypotheses:
                jobs.append(
                    {
                        "instance": instance["name"],
                        "presentation": instance["kind"],
                        "head": head,
                        "window": wi,
                        "hypothesis": hyp,
                        "window_span": [a, b],
                        "path": instance["path"],
                        "n_symbols": instance["n_symbols"],
                        "restarts": restarts.get(head),
                        # wordhom unit-set spec (heads.wordhom.parse_units);
                        # None = the default d5
                        **({"units": units} if head == WORDHOM and units else {}),
                    }
                )
    return jobs


def load_instance(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def arithmetic_stream(inst: dict) -> tuple[np.ndarray, np.ndarray]:
    """Boxer-16 symbols restricted to tokens of length 2–6 (the rung-4
    length range), re-indexed token starts."""
    from ..heads.rung4_arithmetic import SEG_LENGTHS

    sym = np.asarray(inst["symbols"], dtype=np.int64)
    starts = np.asarray(inst["token_starts"], dtype=np.int64)
    ends = np.concatenate([starts[1:], [len(sym)]])
    keep = [(s, e) for s, e in zip(starts, ends) if (e - s) in SEG_LENGTHS]
    ids = np.concatenate([sym[s:e] for s, e in keep])
    st = np.concatenate([[0], np.cumsum([e - s for s, e in keep])[:-1]])
    return ids, st.astype(np.int64)


def instance_record(path: Path) -> dict:
    """Summary used by ``make_jobs`` (no stream copy)."""
    inst = load_instance(path)
    rec = {
        "name": inst["name"],
        "kind": inst["kind"],
        "n_symbols": inst["n_symbols"],
        "n_stream": inst["n_stream"],
        "path": str(path),
        "coverage": inst.get("coverage", {}),
    }
    if inst["kind"] == "boxer" and inst["n_symbols"] == 16:
        _, st = arithmetic_stream(inst)
        rec["n_tokens_kept"] = len(st)
    return rec


# -- solve (forked workers) --------------------------------------------------


def build_ngram_evaluator():
    from ..heads.evaluator import NgramEvaluator
    from ..heads.harness import ngram_calibration_offsets
    from ..heads.ngram import lm_dir, load_lm

    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in LANGS}
    return NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))


def _job_seed(job: dict) -> int:
    return zlib.crc32("/".join(str(job[k]) for k in KEY).encode()) % (2**31)


def solve_job(job: dict) -> dict:
    """One (instance × presentation × head × window × hypothesis) solve on
    the frozen n-gram evaluator → shortlist of keys (+ window decodes)."""
    global _EV, _PARSER
    if _EV is None:
        _EV = build_ngram_evaluator()
    torch.set_num_threads(1)
    t0 = time.time()
    inst = load_instance(job["path"])
    hyp = job["hypothesis"]
    head = job["head"]
    a, b = job["window_span"]
    seed = _job_seed(job)
    out = {k: job[k] for k in KEY}
    out["window_span"] = [a, b]
    cands = []
    if head == "sub1to1":
        from ..heads.rung1_sinkhorn import SinkhornSubstitutionHead

        sym = np.asarray(inst["symbols"][a:b], dtype=np.int64)
        h = SinkhornSubstitutionHead(_EV, seed=seed)
        res = h.solve(sym, language=hyp, restarts=job["restarts"] or 4, shortlist=8)
        for perm, hard, src in res.shortlist:
            cands.append(
                {"map": perm.tolist(), "inner": float(hard), "source": str(src)}
            )
    elif head == "homophonic":
        from ..heads.rung2_homophonic import HomophonicHead

        sym = np.asarray(inst["symbols"][a:b], dtype=np.int64)
        n_sym = int(inst["n_symbols"])
        h = HomophonicHead(_EV, seed=seed)
        rng = np.random.default_rng(seed)
        found = []
        n_restarts = job["restarts"] or 48
        for r in range(n_restarts):
            init = (
                h._frequency_init(sym, n_sym, hyp, rng)
                if r == 0
                else rng.integers(0, A, size=n_sym)
            )
            m, s, _ = h._sa_phase(
                sym,
                np.asarray(init).copy(),
                hyp,
                rng,
                steps=100_000,
                t_start=15.0,
                t_end=0.5,
            )
            found.append((m, float(s), f"sa{r}"))
        seen = set()
        for m, s, src in sorted(found, key=lambda x: -x[1]):
            k = m.tobytes()
            if k in seen:
                continue
            seen.add(k)
            if len(cands) < 4:  # pair-polish the top distinct optima
                m2, s2, _ = h.polish_pairs(sym, m, hyp)
                if s2 > s and m2.tobytes() not in seen:
                    seen.add(m2.tobytes())
                    cands.append(
                        {
                            "map": m2.tolist(),
                            "inner": float(s2),
                            "source": src + "+pairs",
                        }
                    )
            cands.append({"map": m.tolist(), "inner": float(s), "source": src})
            if len(cands) >= 12:
                break
    elif head == "naibbe":
        from ..heads.naibbe_parse import NaibbeParser
        from ..heads.rung3_naibbe import NaibbeBlockHead

        if _PARSER is None:
            _PARSER = NaibbeParser()
        tokens = inst["tokens"][a:b]
        h = NaibbeBlockHead(_EV, _PARSER, seed=seed)
        res = h.solve(tokens, language=hyp, restarts=job["restarts"] or 3, polish=True)
        parses = _PARSER.parse_stream(tokens)
        for maps, score, src in res.shortlist:
            letters, _, _ = h.decode(parses, maps, hyp)
            cands.append(
                {
                    "maps": {f"{k[0]}|{k[1]}": v.tolist() for k, v in maps.items()},
                    "inner": float(score),
                    "source": str(src),
                    "decode": letters.tolist(),
                }
            )
    elif head == "arithmetic":
        from ..heads.rung4_arithmetic import ArithmeticHead

        ids, starts = arithmetic_stream(inst)
        a_c, b_c = int(starts[a]), int(starts[b] if b < len(starts) else len(ids))
        w_ids = ids[a_c:b_c]
        w_starts = starts[a:b] - a_c
        h = ArithmeticHead(_EV, seed=seed)
        res = h.solve_segmented(
            w_ids, w_starts, language=hyp, restarts=job["restarts"] or 3
        )
        for v, u, letters, score, raw_ll3, _rank in res.shortlist:
            cands.append(
                {
                    "v": v.tolist(),
                    "u": u.tolist(),
                    "inner": float(score),
                    "raw_ll3": float(raw_ll3),
                    "source": "restart",
                    "decode": letters.tolist(),
                }
            )
        out["window_chars"] = [a_c, b_c]
    elif head == WORDHOM:
        from ..heads.wordhom import WordHomophonicHead, expand_units, hypothesis_targets

        sym = np.asarray(inst["symbols"][a:b], dtype=np.int64)
        pos = np.asarray(inst["token_pos"][a:b], dtype=np.int64)
        targets = hypothesis_targets(_EV, hyp, units=job.get("units"), inst=inst)
        h = WordHomophonicHead(_EV, seed=seed, targets=targets)
        res = h.solve(
            sym,
            int(inst["n_symbols"]),
            language=hyp,
            token_pos=pos,
            restarts=job["restarts"] or 8,
            sa_steps=job.get("sa_steps") or 100_000,
        )
        for m, score, raw, src in res.shortlist:
            cands.append(
                {
                    "map": m.tolist(),
                    "bigrams": targets.as_list(),
                    "inner": float(score),
                    "raw_ll": float(raw),
                    "source": src,
                    "decode": expand_units(m[sym], targets).tolist(),
                }
            )
        out["violations"] = int(res.violations)
    out["candidates"] = cands
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_solves(
    jobs: list[dict], path: Path, *, workers: int, settings: dict, fresh=False
):
    done = load_done(path, KEY) if not fresh else {}
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to solve", flush=True)
    results = list(done.values())
    # long jobs first so the pool tail is short
    order = {WORDHOM: 0, "naibbe": 0, "arithmetic": 1, "homophonic": 2, "sub1to1": 3}
    todo.sort(key=lambda j: order[j["head"]])

    def dump():
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "settings": settings,
                "instances": results,
            },
        )

    def on_result(i, r, el):
        results.append(r)
        print(
            f"  [{i}/{len(todo)}] {r['instance']} {r['presentation']} {r['head']} w{r['window']} hyp={r['hypothesis']}: "
            f"{len(r['candidates'])} cands ({r['seconds']}s, {el:.0f}s)",
            flush=True,
        )
        dump()

    if todo:
        run_pool(solve_job, todo, workers=workers, on_result=on_result)
    dump()
    return results


# -- score (GPU) -------------------------------------------------------------


def _decode_full(head: str, inst: dict, cand: dict, hyp: str, ng=None, parser=None):
    """Hard decode of the FULL stream under a shortlist key; returns
    (letters, n_cipher_symbols_covered, choice_params)."""
    if head == "sub1to1":
        sym = np.asarray(inst["symbols"], dtype=np.int64)
        m = np.asarray(cand["map"], dtype=np.int64)
        return m[sym], len(sym), {}
    if head == "homophonic":
        sym = np.asarray(inst["symbols"], dtype=np.int64)
        m = np.asarray(cand["map"], dtype=np.int64)
        return m[sym], len(sym), {"sym_to_letter": m}
    if head == "naibbe":
        from ..heads.rung3_naibbe import NaibbeBlockHead

        h = NaibbeBlockHead(ng, parser, seed=0)
        maps = {
            tuple(k.split("|")): np.asarray(v, dtype=np.int64)
            for k, v in cand["maps"].items()
        }
        parses = parser.parse_stream(inst["tokens"])
        letters, _, _ = h.decode(parses, maps, hyp)
        n_chars = sum(len(t) for t in inst["tokens"])
        return letters, n_chars, {"n_tokens": len(inst["tokens"])}
    if head == "arithmetic":
        from ..heads.rung4_arithmetic import ArithmeticHead, segmented_admissible_mask

        ids, starts = arithmetic_stream(inst)
        h = ArithmeticHead(ng, seed=0)
        adm = segmented_admissible_mask(ids, starts)
        _, letters, _ = h.decode_segmented(
            ids, adm, np.asarray(cand["v"]), np.asarray(cand["u"]), language=hyp
        )
        return letters, len(ids), {}
    if head == WORDHOM:
        from ..heads.wordhom import UnitTargets, expand_units

        sym = np.asarray(inst["symbols"], dtype=np.int64)
        m = np.asarray(cand["map"], dtype=np.int64)
        targets = UnitTargets.from_list(cand["bigrams"])
        n_chars = int(inst["coverage"]["n_kept_chars"])
        return (
            expand_units(m[sym], targets),
            n_chars,
            _wordhom_choice_params(inst, m, targets, 0, len(sym)),
        )
    raise ValueError(head)


def _wordhom_choice_params(inst, m, targets, a, b) -> dict:
    from ..heads.wordhom import adjacency, repeat_positions

    sym = np.asarray(inst["symbols"][a:b], dtype=np.int64)
    pos = np.asarray(inst["token_pos"][a:b], dtype=np.int64)
    return {
        "sym_to_unit": m,
        "symbols": sym,
        "repeats": repeat_positions(sym, adjacency(sym, pos)),
        "n_targets": targets.n,
    }


def _choice_total(head: str, decoded: np.ndarray, params: dict) -> float:
    if head == "sub1to1":
        return 0.0
    if head == "homophonic":
        return choice_bits("homophonic", decoded, sym_to_letter=params["sym_to_letter"])
    if head == "naibbe":
        from ..heads.naibbe_parse import NaibbeParser

        return choice_bits(
            "naibbe",
            decoded,
            card_weights=NaibbeParser.CARD_WEIGHTS[False],
            p_unigram=0.476,
            n_tokens=params["n_tokens"],
        )
    if head == "arithmetic":
        return choice_bits("arithmetic", decoded)
    if head == WORDHOM:
        from ..heads.wordhom import choice_bits_total

        return choice_bits_total(
            params["sym_to_unit"],
            params["symbols"],
            params["repeats"],
            params["n_targets"],
        )
    raise ValueError(head)


def _window_decode(head: str, inst: dict, job: dict, cand: dict) -> np.ndarray:
    if head in ("sub1to1", "homophonic"):
        a, b = job["window_span"]
        sym = np.asarray(inst["symbols"][a:b], dtype=np.int64)
        return np.asarray(cand["map"], dtype=np.int64)[sym]
    return np.asarray(cand["decode"], dtype=np.int64)


def score_job(
    ev,
    job: dict,
    inst: dict,
    *,
    budget: int = 64,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    score_windows: int = 16,
    polish_sweeps: int = 6,
    ng=None,
    parser=None,
    polish_choice_term: bool = False,
) -> dict:
    """Outer tier + full-stream scoring for one solve record.

    ``polish_choice_term`` is False by default: the polish runs on the ELBO
    alone (2026-08-25 finding, ``docs/race_polish_plan.md`` §7 — the choice
    term inside the polish destroyed the Borg key). The recorded Phase-6
    runs used True; pass it only to reproduce them."""
    from ..heads.ladder import elbo_polish
    from ..heads.two_tier import Candidate, paired_bits, rescore

    head, hyp = job["head"], job["hypothesis"]
    seed = _job_seed(job)
    t0 = time.time()
    cands = [
        Candidate(
            decode=_window_decode(head, inst, job, c),
            key=c,
            inner_score=c["inner"],
            source=c["source"],
        )
        for c in job["candidates"]
    ]
    rescore(ev, cands, language=hyp, conditions=list(LANGS), n_strata=budget, seed=seed)
    # MDL selection within the job: calibrated plaintext bits + choice bits / char
    offs = ev.calibration_offsets_bits

    def mdl(c: Candidate) -> float:
        params = (
            {"sym_to_letter": np.asarray(c.key["map"])} if head == "homophonic" else {}
        )
        if head == "naibbe":
            a, b = job["window_span"]
            params = {"n_tokens": b - a}
        if head == WORDHOM:
            from ..heads.wordhom import UnitTargets

            a, b = job["window_span"]
            params = _wordhom_choice_params(
                inst,
                np.asarray(c.key["map"], dtype=np.int64),
                UnitTargets.from_list(c.key["bigrams"]),
                a,
                b,
            )
        cb = _choice_total(head, c.decode, params) / max(len(c.decode), 1)
        return calibrate_bits(c.bits[hyp], hyp, offs) + cb

    for c in cands:
        c.extra["mdl"] = mdl(c)
    pick_elbo = min(cands, key=lambda c: c.bits[hyp])
    pick = min(cands, key=lambda c: c.extra["mdl"])
    polish_info = None
    final = pick
    if head == "homophonic":
        a, b = job["window_span"]
        sym = np.asarray(inst["symbols"][a:b], dtype=np.int64)
        m0 = np.asarray(pick.key["map"], dtype=np.int64)

        def choice_fn(m, dec):
            return choice_bits("homophonic", dec, sym_to_letter=m) / max(len(dec), 1)

        m1, polish_info = elbo_polish(
            ev,
            sym,
            m0,
            language=hyp,
            choice_fn=choice_fn if polish_choice_term else None,
            choice_term_in_polish=polish_choice_term,
            sweeps=polish_sweeps,
            budget=8,
            confirm_budget=budget,
            seed=seed,
            pair_swaps=False,
        )
        polish_info = {k: v for k, v in polish_info.items() if k != "trace"} | {
            "n_sweeps": len(polish_info["trace"]),
            "objective": "elbo+choice" if polish_choice_term else "elbo",
        }
        if polish_info["accepted"]:
            c = Candidate(
                decode=m1[sym],
                key={"map": m1.tolist(), "inner": None, "source": "elbo_polish"},
                inner_score=float("nan"),
                source="elbo_polish",
            )
            rescore(
                ev,
                [c],
                language=hyp,
                conditions=list(LANGS),
                n_strata=budget,
                seed=seed,
            )
            c.extra["mdl"] = mdl(c)
            if c.extra["mdl"] < pick.extra["mdl"]:
                final = c
    # full-stream decode + scoring
    letters, n_cipher, params = _decode_full(
        head, inst, final.key, hyp, ng=ng, parser=parser
    )
    letters = np.asarray(letters, dtype=np.int64)
    n_plain = len(letters)
    cb_total = _choice_total(head, letters, params)
    kb = head_key_bits(head, int(inst["n_symbols"]))
    W = ev.window
    cuts = [(s, s + W) for s in range(0, max(n_plain - W + 1, 1), W)]
    if n_plain < W:
        cuts = [(0, n_plain)]
    if len(cuts) > score_windows:
        pick_idx = np.linspace(0, len(cuts) - 1, score_windows).astype(int)
        cuts = [cuts[i] for i in pick_idx]
    rng = np.random.default_rng(seed)
    per_window = []  # [win][seed] -> {"decode": {lang: bits}, "shuffled": {...}}
    for wi, (s, e) in enumerate(cuts):
        dec = letters[s:e]
        shuf = rng.permutation(dec)
        rows = np.stack([dec, shuf])
        entry = {"span": [int(s), int(e)], "seeds": []}
        for si, sd in enumerate(seeds):
            pb = paired_bits(
                ev, rows, list(LANGS), n_strata=budget, seed=seed + 1000 * sd + 17 * wi
            )
            entry["seeds"].append(
                {
                    "decode": {l: float(pb[0, j]) for j, l in enumerate(LANGS)},
                    "shuffled": {l: float(pb[1, j]) for j, l in enumerate(LANGS)},
                }
            )
        per_window.append(entry)
    rec = {k: job[k] for k in KEY}
    rec.update(
        window_span=job["window_span"],
        n_candidates=len(cands),
        shortlist=[
            {
                "source": c.source,
                "inner": c.inner_score,
                "bits": c.bits,
                "mdl": c.extra["mdl"],
            }
            for c in cands
        ],
        pick_elbo_source=pick_elbo.source,
        pick_mdl_source=pick.source,
        elbo_pick_equals_mdl_pick=bool(pick_elbo is pick),
        elbo_polish=polish_info,
        final={
            "source": final.source,
            "key": final.key,
            "window_bits": final.bits,
            "window_mdl": final.extra["mdl"],
            "n_plain_window": len(final.decode),
        },
        full={
            "n_plain": n_plain,
            "n_cipher_covered": int(n_cipher),
            "key_bits": kb,
            "choice_bits": float(cb_total),
            "n_windows_scored": len(cuts),
            "n_windows_total": max((n_plain - W) // W + 1, 1),
            "windows": per_window,
        },
        seconds=round(time.time() - t0, 1),
    )
    return rec


def run_scores(
    solves: list[dict],
    instances: dict[str, dict],
    path: Path,
    *,
    ev,
    budget: int,
    seeds: tuple[int, ...],
    score_windows: int,
    shard: tuple[int, int] = (0, 1),
    fresh: bool = False,
    meta: dict | None = None,
):
    done = load_done(path, KEY) if not fresh else {}
    todo = [
        r
        for i, r in enumerate(solves)
        if tuple(r[k] for k in KEY) not in done and i % shard[1] == shard[0]
    ]
    results = list(done.values())
    ng = parser = None
    if any(r["head"] in ("naibbe", "arithmetic") for r in todo):
        from ..heads.naibbe_parse import NaibbeParser

        ng = build_ngram_evaluator()
        parser = NaibbeParser()
        parser.build_blocks()
    print(f"{len(solves)} solves, {len(done)} scored, {len(todo)} to score", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        inst = instances[(r["instance"], r["presentation"])]
        rec = score_job(
            ev,
            r,
            inst,
            budget=budget,
            seeds=seeds,
            score_windows=score_windows,
            ng=ng,
            parser=parser,
        )
        results.append(rec)
        wb = rec["final"]["window_bits"]
        print(
            f"  [{i}/{len(todo)}] {r['instance']} {r['presentation']} {r['head']} w{r['window']} hyp={r['hypothesis']}: "
            + " ".join(f"{l[:2]}={wb[l]:.3f}" for l in LANGS)
            + f" final={rec['final']['source']} ({rec['seconds']}s, {time.time()-t0:.0f}s)",
            flush=True,
        )
        write_json_atomic(
            path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                **(meta or {}),
                "instances": results,
            },
        )
    write_json_atomic(
        path,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            **(meta or {}),
            "instances": results,
        },
    )
    return results


# -- report ------------------------------------------------------------------


def order0_entropy_bits(symbols: np.ndarray) -> float:
    c = np.bincount(np.asarray(symbols)).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def cell_from_score(rec: dict, table: CalibrationTable, inst_meta: dict) -> dict:
    """Collapse one scored record into the reported cell."""
    offs = table.additive_offsets()
    hyp = rec["hypothesis"]
    wins = rec["full"]["windows"]
    # per (window, seed) calibrated bits under every condition
    dec = {
        l: np.array(
            [
                [calibrate_bits(s["decode"][l], l, offs) for s in w["seeds"]]
                for w in wins
            ]
        )
        for l in LANGS
    }
    shuf = {
        l: np.array(
            [
                [calibrate_bits(s["shuffled"][l], l, offs) for s in w["seeds"]]
                for w in wins
            ]
        )
        for l in LANGS
    }
    n_w, n_s = dec[hyp].shape
    own = dec[hyp]
    mean = float(own.mean())
    win_means = own.mean(axis=1)
    sem = float(win_means.std(ddof=1) / np.sqrt(n_w)) if n_w > 1 else 0.0
    # language rank of the decode from the full-stream means, per seed (flip-rate)
    by_lang = {l: float(dec[l].mean()) for l in LANGS}
    ranking = sorted(by_lang, key=by_lang.get)
    top = ranking[0]
    margin = by_lang[ranking[1]] - by_lang[ranking[0]]
    seed_tops = [min(LANGS, key=lambda l: dec[l][:, si].mean()) for si in range(n_s)]
    flip = float(np.mean([t != top for t in seed_tops]))
    win_tops = [min(LANGS, key=lambda l: dec[l][wi].mean()) for wi in range(n_w)]
    win_vote = float(np.mean([t == top for t in win_tops]))
    structure = float((shuf[hyp] - dec[hyp]).mean())
    structure_all = {l: float((shuf[l] - dec[l]).mean()) for l in LANGS}
    n_plain = rec["full"]["n_plain"]
    n_cov = rec["full"]["n_cipher_covered"]
    kb, cb = rec["full"]["key_bits"], rec["full"]["choice_bits"]
    total = mean * n_plain + kb + cb
    n_all = inst_meta.get("n_cipher_all", n_cov)
    # Uncovered ciphertext symbols are charged at what the best NO-CIPHER
    # description of this stream costs (its own held-out n-gram
    # cross-entropy; order-0 entropy if the baselines are absent) — a head
    # that explains only part of the manuscript must pay the surface price
    # for the rest, or partial coverage would look like compression.
    base = inst_meta.get("no_cipher_baselines") or {}
    ngram_bits = [v for k, v in base.items() if k.startswith("ngram")]
    uncovered_rate = (
        min(ngram_bits) if ngram_bits else inst_meta.get("order0_entropy_bits", 0.0)
    )
    total_all = total + (n_all - n_cov) * uncovered_rate
    language_like = bool(
        mean <= ABSTAIN_RULE["max_plain_bits"]
        and structure >= ABSTAIN_RULE["min_structure_margin"]
    )
    return {
        **{k: rec[k] for k in KEY},
        "window_span": rec["window_span"],
        "final_source": rec["final"]["source"],
        "elbo_pick_equals_mdl_pick": rec.get("elbo_pick_equals_mdl_pick"),
        "elbo_polish_accepted": (rec.get("elbo_polish") or {}).get("accepted"),
        "plain_bits": mean,
        "plain_bits_sem": sem,
        "plain_bits_window_sd": float(win_means.std(ddof=1)) if n_w > 1 else 0.0,
        "plain_bits_by_condition": by_lang,
        "window_bits_by_condition": {
            l: calibrate_bits(rec["final"]["window_bits"][l], l, offs) for l in LANGS
        },
        "n_plain": n_plain,
        "n_cipher_covered": n_cov,
        "n_cipher_all": n_all,
        "coverage": n_cov / max(n_all, 1),
        "key_bits": kb,
        "choice_bits": cb,
        "choice_bits_per_plain": cb / max(n_plain, 1),
        "penalty_per_plain": (kb + cb) / max(n_plain, 1),
        "total_per_covered_symbol": total / max(n_cov, 1),
        "total_per_all_symbols": total_all / max(n_all, 1),
        "uncovered_charge_bits_per_symbol": float(uncovered_rate),
        "no_cipher_baseline_bits_per_symbol": float(uncovered_rate),
        "structure_margin": structure,
        "structure_margin_by_condition": structure_all,
        "language_rank_of_decode": ranking,
        "top_language_of_decode": top,
        "top_margin_bits": float(margin),
        "top_margin_uncertainty_bits": table.margin_uncertainty_bits(
            ranking[0], ranking[1]
        ),
        "replicate_flip_rate": flip,
        "window_vote_for_top": win_vote,
        "n_windows": n_w,
        "n_seeds": n_s,
        "language_like": language_like,
    }


def rank_table(cells: list[dict]) -> dict:
    """The (cipher × language) table of one stream: for every head the best
    cell per hypothesis (lowest total per covered symbol over presentations
    and windows), ranked; agreement statistics across windows /
    presentations / heads."""
    by_head: dict[str, dict[str, list[dict]]] = {}
    for c in cells:
        by_head.setdefault(c["head"], {}).setdefault(c["hypothesis"], []).append(c)
    rows = []
    for head, hyps in by_head.items():
        for hyp, cs in hyps.items():
            best = min(cs, key=lambda c: c["total_per_all_symbols"])
            rows.append(
                {
                    "head": head,
                    "hypothesis": hyp,
                    "n_cells": len(cs),
                    "best": best,
                    "total_per_all_symbols": best["total_per_all_symbols"],
                    "total_per_covered_symbol": best["total_per_covered_symbol"],
                    "beats_no_cipher_baseline": bool(
                        best["total_per_all_symbols"]
                        < best.get("no_cipher_baseline_bits_per_symbol", float("inf"))
                    ),
                    "total_spread_over_cells": float(
                        max(c["total_per_all_symbols"] for c in cs)
                        - min(c["total_per_all_symbols"] for c in cs)
                    ),
                    "plain_bits": best["plain_bits"],
                    "language_like_any": any(c["language_like"] for c in cs),
                }
            )
    rows.sort(key=lambda r: r["total_per_all_symbols"])
    # within-head language ranking + agreement
    per_head = {}
    for head, hyps in by_head.items():
        order = sorted(
            hyps, key=lambda h: min(c["total_per_all_symbols"] for c in hyps[h])
        )
        tops = {}
        for hyp, cs in hyps.items():
            for c in cs:
                tops[(c["presentation"], c["window"], hyp)] = c["total_per_all_symbols"]
        # agreement: for each (presentation, window) the argmin hypothesis
        groups: dict[tuple, dict] = {}
        for (pres, w, hyp), v in tops.items():
            groups.setdefault((pres, w), {})[hyp] = v
        votes = [min(g, key=g.get) for g in groups.values() if len(g) == len(LANGS)]
        per_head[head] = {
            "language_order": order,
            "top": order[0],
            "margin_top2_bits_per_symbol": (
                float(
                    min(c["total_per_all_symbols"] for c in hyps[order[1]])
                    - min(c["total_per_all_symbols"] for c in hyps[order[0]])
                )
                if len(order) > 1
                else None
            ),
            "votes_by_presentation_window": votes,
            "agreement": (
                float(np.mean([v == order[0] for v in votes])) if votes else None
            ),
            "n_votes": len(votes),
        }
    head_tops = [v["top"] for v in per_head.values()]
    return {
        "ranked": rows,
        "per_head": per_head,
        "head_agreement_on_top_language": (
            float(np.mean([t == rows[0]["hypothesis"] for t in head_tops]))
            if rows
            else None
        ),
        "abstain": not any(r["language_like_any"] for r in rows),
    }


def fmt_table_md(name: str, tab: dict, cells: list[dict]) -> str:
    md = [
        f"#### {name}",
        "",
        "| rank | head | language | MDL total / ciphertext symbol | / covered symbol | no-cipher baseline | plaintext bits/char ± sem | key bits | choice bits/char | coverage | structure margin | flip-rate | window vote | language-like |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(tab["ranked"], 1):
        b = r["best"]
        md.append(
            f"| {i} | {r['head']} | {r['hypothesis']} | {r['total_per_all_symbols']:.3f} | {r['total_per_covered_symbol']:.3f} | {b['no_cipher_baseline_bits_per_symbol']:.3f}{' ✓' if r['beats_no_cipher_baseline'] else ''} | {b['plain_bits']:.3f} ± {b['plain_bits_sem']:.3f} | {b['key_bits']:.0f} | {b['choice_bits_per_plain']:.2f} | {b['coverage']:.2f} | {b['structure_margin']:.2f} | {b['replicate_flip_rate']:.2f} | {b['window_vote_for_top']:.2f} | {'yes' if b['language_like'] else 'no'} |"
        )
    md.append("")
    md.append(
        "per head: "
        + "; ".join(
            f"{h}: {v['language_order']} (margin {v['margin_top2_bits_per_symbol']:.3f} bits/symbol, agreement {v['agreement']} of {v['n_votes']})"
            for h, v in tab["per_head"].items()
        )
    )
    md.append(
        f"head agreement on top language: {tab['head_agreement_on_top_language']}; abstain: {tab['abstain']}"
    )
    return "\n".join(md)


def ciphertext_baselines(
    symbols: np.ndarray, n_symbols: int, orders=(1, 2, 3, 4, 5)
) -> dict:
    """The no-cipher description of the ciphertext: order-0 entropy and the
    held-out cross-entropy (bits/symbol) of order-k symbol n-gram models
    fitted on the first half of the stream (add-0.5 smoothing with backoff
    to the lower order), evaluated on the second half. A (cipher × language)
    cell whose MDL total per symbol is above these numbers explains the
    ciphertext WORSE than its own surface statistics — the floor a
    decipherment claim must clear."""
    sym = np.asarray(symbols, dtype=np.int64)
    n = len(sym)
    half = n // 2
    train, test = sym[:half], sym[half:]
    out = {"order0_entropy_bits": order0_entropy_bits(sym), "n_symbols": int(n_symbols)}
    # counts of contexts up to the max order on the train half
    from collections import defaultdict

    K = max(orders)
    counts = [defaultdict(lambda: np.zeros(n_symbols)) for _ in range(K + 1)]
    for i in range(len(train)):
        for k in range(K + 1):
            if i - k < 0:
                break
            ctx = tuple(train[i - k : i].tolist())
            counts[k][ctx][train[i]] += 1
    uni = counts[0][()]

    def prob(k, ctx, s):
        if k == 0:
            return (uni[s] + 0.5) / (uni.sum() + 0.5 * n_symbols)
        c = counts[k].get(ctx)
        if c is None:
            return prob(k - 1, ctx[1:], s)
        tot = c.sum()
        lam = tot / (tot + n_symbols)  # Witten-Bell style backoff weight
        return lam * (c[s] / tot) + (1 - lam) * prob(k - 1, ctx[1:], s)

    for k in orders:
        ll = 0.0
        for i in range(k, len(test)):
            ctx = tuple(test[i - k : i].tolist())
            ll += -np.log2(prob(k, ctx, test[i]))
        out[f"ngram{k}_heldout_bits"] = float(ll / max(len(test) - k, 1))
    return out
