"""Task 3.1 acceptance — common random numbers across language conditions.

Scores a fixed set of held-out windows under every conditioning language
``R`` times with fresh masking seeds, once with CRN (every condition sees the
same masks) and once with independent masks per condition, and compares the
replicate-to-replicate variance of the *between-language score differences*
— the quantity the ranking's argmax consumes. Acceptance: variance under CRN
≥ 5× smaller than under independent sampling.

Also reported: the variance of the raw per-condition score (unchanged by CRN
— it is the shared noise that cancels), and the same experiment at a short
window length where the Monte-Carlo noise matters most.

Usage:
    uv run python scripts/crn_check.py --ckpt DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt
Writes DATA_ROOT/analysis/phase3/crn_check.json (+ ClearML task ``task3.1``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.infra.checkpoint import load_backbone
from diff_voyn.metrology import ScoreSettings, score_conditions

LANGS = tuple(LANG_TO_INDEX)
ACCEPT_RATIO = 5.0


def select_windows(
    heldout: CorpusWindows, lang: str, n: int, length: int
) -> np.ndarray:
    tiled = heldout.tiled_windows(lang, 1024)
    pick = np.linspace(0, len(tiled) - 1, num=min(n, len(tiled))).round().astype(int)
    return tiled[pick, :length]


def run(model, ids: np.ndarray, *, strata: int, reps: int, batch: int, device: str):
    """Returns {"crn": [R, N, C], "independent": [R, N, C]}."""
    out = {}
    for mode, crn in (("crn", True), ("independent", False)):
        runs = []
        for r in range(reps):
            st = ScoreSettings(n_strata=strata, seed=10_000 * (r + 1), batch=batch)
            runs.append(
                score_conditions(model, ids, LANGS, settings=st, device=device, crn=crn)
            )
        out[mode] = np.stack(runs)
    return out


def analyze(scores: dict[str, np.ndarray]) -> dict:
    res = {"pairs": {}, "raw": {}}
    ratios = []
    for a, b in itertools.combinations(range(len(LANGS)), 2):
        key = f"{LANGS[a]}-{LANGS[b]}"
        entry = {}
        for mode, arr in scores.items():
            diff = arr[:, :, a] - arr[:, :, b]  # [R, N]
            var = diff.var(axis=0, ddof=1)  # across replicates, per window
            entry[mode] = {
                "var_mean": float(var.mean()),
                "sd_mean": float(np.sqrt(var).mean()),
                "diff_mean": float(diff.mean()),
            }
        ratio = entry["independent"]["var_mean"] / entry["crn"]["var_mean"]
        entry["variance_ratio_independent_over_crn"] = float(ratio)
        ratios.append(ratio)
        res["pairs"][key] = entry
    for j, lang in enumerate(LANGS):
        res["raw"][lang] = {
            mode: float(arr[:, :, j].var(axis=0, ddof=1).mean())
            for mode, arr in scores.items()
        }
    res["min_variance_ratio"] = float(min(ratios))
    res["pass"] = bool(min(ratios) >= ACCEPT_RATIO)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_b-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--windows", type=int, default=32, help="per language")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--strata", type=int, default=32)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 200])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            l: [d["doc_id"] for d in sp["heldout"]]
            for l, sp in splits["languages"].items()
        },
    )
    model, meta = load_backbone(args.ckpt, args.device)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "3.1",
        "backbone": meta,
        "settings": vars(args) | {"ckpt": str(args.ckpt)},
        "acceptance": f"variance of between-language differences: independent/CRN ≥ {ACCEPT_RATIO}",
        "by_length": {},
    }
    out_dir = root / "analysis" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for length in args.lengths:
        ids = np.concatenate(
            [select_windows(heldout, l, args.windows, length) for l in LANGS]
        )
        t0 = time.time()
        scores = run(
            model,
            ids,
            strata=args.strata,
            reps=args.reps,
            batch=args.batch,
            device=args.device,
        )
        res = analyze(scores)
        res["n_windows"] = len(ids)
        res["seconds"] = round(time.time() - t0, 1)
        report["by_length"][str(length)] = res
        for mode, arr in scores.items():
            arrays[f"L{length}/{mode}"] = arr
        print(
            f"L={length}: min variance ratio {res['min_variance_ratio']:.1f} "
            f"({'PASS' if res['pass'] else 'FAIL'})  {res['seconds']:.0f}s",
            flush=True,
        )
        for k, v in res["pairs"].items():
            print(
                f"   {k:16s} sd(diff) CRN {v['crn']['sd_mean']:.4f}  indep "
                f"{v['independent']['sd_mean']:.4f}  ratio {v['variance_ratio_independent_over_crn']:.1f}"
            )
    report["pass"] = all(r["pass"] for r in report["by_length"].values())
    out = out_dir / "crn_check.json"
    out.write_text(json.dumps(report, indent=2))
    np.savez(out_dir / "crn_check_scores.npz", **arrays)
    print(f"overall: {'PASS' if report['pass'] else 'FAIL'}; written {out}")

    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name="crn-check", phase="phase3"), root, tags=["task3.1"]
        )
        task.connect_configuration(report, name="crn_check")
        logger = task.get_logger()
        for length, res in report["by_length"].items():
            for k, v in res["pairs"].items():
                logger.report_scalar(
                    "crn_variance_ratio",
                    f"{k}@L{length}",
                    v["variance_ratio_independent_over_crn"],
                    0,
                )
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
