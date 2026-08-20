"""Gate G1 verification (end of Phase 1).

Checks, in order:

1. **Plateau (task 1.4):** per-language held-out canary from the Phase-A run
   logs — relative improvement over the trailing 1000 steps, criterion <0.5%.
   Because the canary is defined on EMA weights (decay 0.9999 ⇒ ~10k-step lag),
   the final checkpoints are also scored with raw vs EMA weights: a large
   raw-vs-EMA gap means residual canary slope is EMA catch-up, not model
   improvement.
2. **Interference (task 1.5):** no language stagnating while others improve —
   trailing-half improvement reported per language; any language below half
   the median improvement of the others is flagged.
3. **Ranking agreement (task 1.6):** 25M and 85M (final EMA weights) rank
   clean held-out windows by per-window conditional NELBO under common random
   numbers; top-1 agreement rate between the two models is reported, along
   with each model's top-1-equals-true-language rate.
4. **Calibration table v1 (task 3.4, G1 checklist):** per-language held-out
   NELBO alongside the v1 n-gram AR reference NLL, written to
   ``DATA_ROOT/calibration/calibration_v1.json``. The n-gram is a *provisional*
   AR reference — the design-§5b.3 small char-AR transformer offsets are an
   upgrade item (needs GPU); offsets are recorded but flagged provisional.

Run: ``uv run python scripts/g1_check.py [--no-clearml] [--device cpu]``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows
from diff_voyn.infra.config import ModelConfig
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char, per_window_nelbo_bits
from diff_voyn.model.backbone import Backbone

RUNS = {"85m": "phase_a-85m-seed0", "25m": "phase_a-25m-seed0"}
LOGS = {"85m": "phase_a-85m.log", "25m": "phase_a-25m.log"}
EVAL_RE = re.compile(r"step\s+(\d+)\s+heldout NELBO \(EMA, cond\|uncond\): (.*)$")
LANG_RE = re.compile(r"(\w+) ([\d.]+)\|([\d.]+)u")

PLATEAU_WINDOW_STEPS = 1000
PLATEAU_THRESHOLD = 0.005

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        (WARNINGS if warn_only else FAILURES).append(name)


def parse_canary(log_path: Path) -> dict[int, dict[str, tuple[float, float]]]:
    """step -> {lang: (cond_bits, uncond_bits)}."""
    series: dict[int, dict[str, tuple[float, float]]] = {}
    for line in log_path.read_text().splitlines():
        m = EVAL_RE.search(line)
        if m:
            series[int(m.group(1))] = {
                lang: (float(c), float(u)) for lang, c, u in LANG_RE.findall(m.group(2))
            }
    return series


def load_models(ckpt_path: Path, device: str) -> tuple[Backbone, Backbone, int]:
    """(raw, ema, step) from one checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["extra"]["config"]["model"])
    raw = Backbone(cfg).to(device).eval()
    raw.load_state_dict(state["model"])
    ema = Backbone(cfg).to(device).eval()
    ema.load_state_dict(state["model"])
    sd = ema.state_dict()
    for k, v in state["ema"]["shadow"].items():
        sd[k].copy_(v.to(sd[k].dtype))
    return raw, ema, state["step"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strata", type=int, default=16)
    p.add_argument("--probe-windows", type=int, default=8, help="per language")
    p.add_argument("--probe-len", type=int, default=512)
    args = p.parse_args()
    root = data_root()
    report: dict = {"created_utc": datetime.now(UTC).isoformat()}

    # ------------------------------------------------------------------ G1.1
    print(
        "G1.1 per-language canary plateau (EMA, trailing "
        f"{PLATEAU_WINDOW_STEPS} steps, criterion <{PLATEAU_THRESHOLD:.1%})"
    )
    canary: dict[str, dict] = {}
    for size, log_name in LOGS.items():
        series = parse_canary(root / "runs" / log_name)
        steps = sorted(series)
        last = steps[-1]
        ref = max(s for s in steps if s <= last - PLATEAU_WINDOW_STEPS)
        canary[size] = {"final_step": last, "series": series}
        for lang in LANG_TO_INDEX:
            v0, v1 = series[ref][lang][0], series[last][lang][0]
            rel = (v0 - v1) / v0
            check(
                f"{size} {lang}: {v0:.3f}→{v1:.3f} over steps {ref}→{last} "
                f"({rel:+.2%})",
                rel < PLATEAU_THRESHOLD,
                warn_only=True,  # verdict refined by the EMA-lag check below
            )
            report.setdefault("plateau", {})[f"{size}/{lang}"] = {
                "ref_step": ref,
                "ref": v0,
                "final": v1,
                "trailing_rel": rel,
            }

    # Raw-vs-EMA gap on the final checkpoints (EMA-lag diagnosis).
    print(
        "G1.1b raw vs EMA held-out NELBO on final checkpoints "
        "(is residual slope EMA catch-up?)"
    )
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout_ids = {
        lang: [d["doc_id"] for d in sp["heldout"]]
        for lang, sp in splits["languages"].items()
    }
    windows = CorpusWindows(corpus_dir, heldout_ids)
    rng = np.random.default_rng(12345)
    eval_batch = {
        lang: torch.from_numpy(
            np.stack([windows.sample_window(lang, 1024, rng) for _ in range(6)]).astype(
                np.int64
            )
        )
        for lang in LANG_TO_INDEX
    }
    ema_models: dict[str, Backbone] = {}
    for size, run in RUNS.items():
        raw, ema, step = load_models(root / "runs" / run / "ckpt_final.pt", args.device)
        ema_models[size] = ema
        for lang, ids in eval_batch.items():
            common = {"n_strata": args.strata, "seed": 0, "device": args.device}
            b_raw = estimate_nelbo_bits_per_char(
                raw, ids, LANG_TO_INDEX[lang], **common
            )
            b_ema = estimate_nelbo_bits_per_char(
                ema, ids, LANG_TO_INDEX[lang], **common
            )
            gap = (b_ema - b_raw) / b_raw
            print(
                f"  {size} {lang}: raw {b_raw:.3f}  ema {b_ema:.3f}  " f"gap {gap:+.2%}"
            )
            report.setdefault("raw_vs_ema", {})[f"{size}/{lang}"] = {
                "raw": b_raw,
                "ema": b_ema,
                "rel_gap": gap,
                "step": step,
            }
        del raw

    # ------------------------------------------------------------------ G1.2
    print("G1.2 interference watch (trailing-half improvement per language)")
    for size in RUNS:
        series = canary[size]["series"]
        steps = sorted(series)
        half = steps[len(steps) // 2]
        last = steps[-1]
        improvements = {
            lang: (series[half][lang][0] - series[last][lang][0])
            / series[half][lang][0]
            for lang in LANG_TO_INDEX
        }
        med = float(np.median(list(improvements.values())))
        for lang, imp in improvements.items():
            check(
                f"{size} {lang}: {imp:+.2%} since step {half} (median {med:+.2%})",
                imp > 0.5 * med,
            )
        report.setdefault("interference", {})[size] = improvements

    # ------------------------------------------------------------------ G1.3
    print("G1.3 25M/85M ranking-agreement probe on clean held-out windows")
    probe_rng = np.random.default_rng(777)
    probe_ids, probe_true = [], []
    for lang in LANG_TO_INDEX:
        for _ in range(args.probe_windows):
            probe_ids.append(windows.sample_window(lang, args.probe_len, probe_rng))
            probe_true.append(LANG_TO_INDEX[lang])
    ids = torch.from_numpy(np.stack(probe_ids).astype(np.int64))
    true = np.array(probe_true)
    scores = {}
    for size, model in ema_models.items():
        cols = [
            per_window_nelbo_bits(
                model,
                ids,
                LANG_TO_INDEX[lang],
                n_strata=args.strata,
                seed=0,
                device=args.device,
            ).numpy()
            for lang in LANG_TO_INDEX  # CRN: same seed for every condition
        ]
        scores[size] = np.stack(cols, axis=1)  # [n_windows, n_lang]
    top1 = {size: s.argmin(axis=1) for size, s in scores.items()}
    n = len(true)
    agree = float((top1["85m"] == top1["25m"]).mean())
    order_agree = float(
        (np.argsort(scores["85m"], axis=1) == np.argsort(scores["25m"], axis=1))
        .all(axis=1)
        .mean()
    )
    check(f"top-1 agreement 85m vs 25m: {agree:.0%} ({n} windows)", agree >= 0.9)
    print(f"  full-ranking agreement: {order_agree:.0%}")
    for size in RUNS:
        acc = float((top1[size] == true).mean())
        check(
            f"{size} top-1 = true language on clean text: {acc:.0%}",
            acc >= 0.9,
            warn_only=True,  # formally a Phase-3 (3.6) bar, tracked early
        )
        report.setdefault("ranking_probe", {})[size] = {
            "top1_true_rate": acc,
            "scores": scores[size].tolist(),
        }
    report["ranking_probe"]["top1_agreement"] = agree
    report["ranking_probe"]["order_agreement"] = order_agree
    report["ranking_probe"]["true_lang_idx"] = true.tolist()

    # ------------------------------------------------------------------ G1.4
    print("G1.4 calibration table v1 (provisional n-gram AR reference)")
    ngram = json.loads((root / "ngram_lms" / "v1" / "summary.json").read_text())
    table = {}
    for lang in LANG_TO_INDEX:
        nelbo = canary["85m"]["series"][canary["85m"]["final_step"]][lang][0]
        nll_ar = ngram[lang]["heldout_bits_per_char"]
        table[lang] = {
            "heldout_nelbo_bits_85m_ema": nelbo,
            "nll_ar_bits_ngram5": nll_ar,
            "offset_bits_provisional": round(nelbo - nll_ar, 4),
        }
        print(
            f"  {lang}: NELBO {nelbo:.3f}  ngram5 NLL {nll_ar:.3f}  "
            f"offset {nelbo - nll_ar:+.3f} (provisional)"
        )
    cal_dir = root / "calibration"
    cal_dir.mkdir(exist_ok=True)
    cal = {
        "calibration_version": "v1",
        "created_utc": report["created_utc"],
        "phase": "phase_a",
        "checkpoint": str(root / "runs" / RUNS["85m"] / "ckpt_final.pt"),
        "reference": "ngram v1 (order-5, train split) — PROVISIONAL; "
        "design §5b.3 char-AR transformer offsets pending (GPU required). "
        "NELBO−NLL_ngram mixes bound gap with n-gram model deficiency; "
        "do not apply as a ranking offset until the AR-transformer version "
        "replaces it.",
        "languages": table,
    }
    (cal_dir / "calibration_v1.json").write_text(json.dumps(cal, indent=2))
    print(f"  written: {cal_dir / 'calibration_v1.json'}")
    report["calibration_v1"] = cal

    (root / "runs" / "g1_report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {root / 'runs' / 'g1_report.json'}")

    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        cfg = RunConfig(run_name="g1-check", phase="phase1")
        task = init_task(cfg, root, tags=["g1"])
        task.connect_configuration(report, name="g1_report")
        logger = task.get_logger()
        for key, d in report["raw_vs_ema"].items():
            logger.report_scalar("g1_final_heldout_nelbo_raw", key, d["raw"], 0)
            logger.report_scalar("g1_final_heldout_nelbo_ema", key, d["ema"], 0)
        logger.report_scalar("g1_ranking_probe", "top1_agreement", agree, 0)
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()

    print()
    if WARNINGS:
        print(f"G1: {len(WARNINGS)} warning(s): {WARNINGS}")
    if FAILURES:
        print(f"G1: {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("G1: hard checks passed (see warnings for gate judgement)")


if __name__ == "__main__":
    main()
