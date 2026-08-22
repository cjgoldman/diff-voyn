"""Per-language bound calibration — task 3.4 (design §5b), calibration table v1.

For every frozen language, on the *same* text (the full tiled held-out split,
1024-char windows):

- ``NELBO``  — diffusion backbone (EMA weights), per-window Rao-Blackwellized
  NELBO, stratified timesteps with common random numbers across language
  conditions (all three conditioning languages + unconditional are scored, so
  the same run feeds the fairness audit 3.5 and the per-window LID accuracy);
- ``NLL_AR`` — the §5b.3 char-AR reference model of that language
  (``scripts/train_ar_reference.py``), exact per-window cross-entropy.

``offset = NELBO − NLL_AR`` (bits/char, own-language condition) is the bound-gap
estimate. It is stored here, versioned, and applied in exactly one place
(``EvaluatorBase.calibrated_bits_per_char`` — task 3.4 "single-sourced").

Reference tiers: ``--ar-dir`` may hold per-language models
(``<dir>/<lang>/ar_best.pt``, v1/v2) or ONE multilingual language-conditioned
model (``<dir>/multilingual/ar_best.pt``, v3 — scored under each language's
own condition). The multilingual tier removes the per-language data-starvation
confound of the monolingual references (see ``docs/phase3_status.md``).

Usage:
    uv run python scripts/calibrate.py [--ckpt DATA_ROOT/runs/phase_a-85m-seed0/ckpt_final.pt]
                                       [--ar-which best|final] [--strata 32] [--version v1]
    uv run python scripts/calibrate.py --ckpt .../phase_b-85m-seed0/ckpt_final.pt \
        --ar-dir DATA_ROOT/ar_reference/v3 --phase phase_b --version v3

Writes DATA_ROOT/calibration/calibration_<version>.json and the per-window
arrays (+ window→document index) in calibration_<version>_windows.npz.
"""

from __future__ import annotations

import argparse
import json
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
from diff_voyn.metrology.scoring import (
    DEFAULT_CONDITIONS,
    ScoreSettings,
    score_conditions,
)
from diff_voyn.model.ar_reference import ARConfig, CharARLM
from diff_voyn.model.backbone import Backbone

LN2 = float(np.log(2.0))


def load_ema_backbone(ckpt: Path, device: str) -> tuple[Backbone, dict]:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["extra"]["config"]["model"])
    model = Backbone(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    sd = model.state_dict()
    for k, v in state["ema"]["shadow"].items():
        sd[k].copy_(v.to(sd[k].dtype))
    meta = {
        "path": str(ckpt),
        "step": state["step"],
        "ema_decay": state["ema"]["decay"],
        "schedule": state["extra"].get("schedule"),
        "model": state["extra"]["config"]["model"],
    }
    return model, meta


def load_ar(path: Path, device: str) -> tuple[CharARLM, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = CharARLM(ARConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    return model, {k: v for k, v in state.items() if k != "model"}


def score_diffusion(
    model: Backbone,
    ids: torch.Tensor,
    *,
    strata: int,
    seed: int,
    batch: int,
    device: str,
) -> np.ndarray:
    """[n_windows, n_lang + 1] bits/char: columns = frozen language order,
    then unconditional. CRN: every condition of a chunk shares one seed
    (``metrology.score_conditions``; chunk ``i`` uses ``seed + i`` exactly as
    the v1 table did, so tables are paired window-for-window)."""
    return score_conditions(
        model,
        ids,
        DEFAULT_CONDITIONS,
        settings=ScoreSettings(n_strata=strata, seed=seed, batch=batch),
        device=device,
    )


@torch.no_grad()
def score_ar(
    model: CharARLM,
    ids: torch.Tensor,
    *,
    batch: int,
    device: str,
    lang_idx: int | None = None,
) -> np.ndarray:
    out = []
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch].to(device)
        lang = (
            None
            if lang_idx is None
            else torch.full((len(chunk),), lang_idx, dtype=torch.long, device=device)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model.nll_bits_per_char(chunk, lang).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def load_reference(ar_dir: Path, which: str, device: str):
    """Returns ``(get_model(lang) -> (model, meta), tier_description)``."""
    ml = ar_dir / "multilingual" / f"ar_{which}.pt"
    if ml.exists():
        model, meta = load_ar(ml, device)
        if not model.cfg.multilingual:
            raise ValueError(f"{ml} is not a multilingual reference")
        return (lambda lang: (model, meta)), (
            "multilingual language-conditioned char-AR (one model, backbone's "
            f"τ-balanced mix, design §5b.3), {ar_dir.name}/'{which}' checkpoint"
        )
    cache = {}

    def get(lang):
        if lang not in cache:
            cache[lang] = load_ar(ar_dir / lang / f"ar_{which}.pt", device)
        return cache[lang]

    return get, (
        "char-AR transformer per language (design §5b.3), "
        f"scripts/train_ar_reference.py, {ar_dir.name}/'{which}' checkpoints"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_a-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--ar-dir", type=Path, default=root / "ar_reference" / "v1")
    p.add_argument("--ar-which", choices=["best", "final"], default="best")
    p.add_argument("--strata", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--version", default="v1")
    p.add_argument("--phase", default="phase_a")
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="debug cap on windows per language (recorded in the output; the "
        "real table must be produced uncapped)",
    )
    p.add_argument(
        "--derive-report-only",
        metavar="SOURCE_VERSION",
        default=None,
        help="write --version as a copy of SOURCE_VERSION under the report-only "
        "policy (applied offsets zero, measured offsets kept); no scoring",
    )
    p.add_argument(
        "--nelbo-from",
        metavar="VERSION",
        default=None,
        help="reuse the per-window diffusion NELBO arrays of an existing table "
        "(same backbone, windows, seeds, strata) and score only the AR reference "
        "— a reference swap without GPU work; the backbone metadata is copied",
    )
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.derive_report_only:
        from diff_voyn.metrology.calibration import derive_report_only

        out = derive_report_only(args.derive_report_only, args.version, root)
        print(
            f"written {out} (report-only policy, derived from {args.derive_report_only})"
        )
        return
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            lang: [d["doc_id"] for d in sp["heldout"]]
            for lang, sp in splits["languages"].items()
        },
    )
    reuse = None
    if args.nelbo_from:
        src = json.loads(
            (root / "calibration" / f"calibration_{args.nelbo_from}.json").read_text()
        )
        reuse = np.load(
            root / "calibration" / f"calibration_{args.nelbo_from}_windows.npz"
        )
        bb_meta = src["backbone"]
        args.ckpt = Path(bb_meta["path"])
        args.strata, args.seed = src["scoring"]["strata"], src["scoring"]["seed"]
        if src["scoring"].get("max_windows_cap"):
            raise SystemExit("source table was capped; cannot reuse")
        backbone = None
        print(f"reusing diffusion NELBO arrays of calibration {args.nelbo_from}")
    else:
        backbone, bb_meta = load_ema_backbone(args.ckpt, args.device)
    seq_len = bb_meta["model"]["seq_len"]
    print(
        f"backbone {args.ckpt.name} step {bb_meta['step']} (EMA {bb_meta['ema_decay']})"
    )

    get_reference, reference_desc = load_reference(
        args.ar_dir, args.ar_which, args.device
    )
    table, arrays, ar_meta = {}, {}, {}
    for lang, li in LANG_TO_INDEX.items():
        tiled, doc_index = heldout.tiled_windows_by_doc(lang, seq_len)
        ids = torch.from_numpy(tiled.astype(np.int64))
        if args.max_windows:
            ids = ids[: args.max_windows]
            doc_index = doc_index[: args.max_windows]
        ar, meta = get_reference(lang)
        ar_meta[lang] = dict(meta) | {"language": lang}
        if isinstance(meta.get("heldout_bits_per_char"), dict):
            ar_meta[lang]["heldout_bits_per_char"] = meta["heldout_bits_per_char"][lang]
        if reuse is not None:
            nelbo = reuse[f"{lang}/nelbo"]
            if len(nelbo) != len(ids):
                raise SystemExit(
                    f"{lang}: {len(nelbo)} reused windows vs {len(ids)} tiled"
                )
        else:
            nelbo = score_diffusion(
                backbone,
                ids,
                strata=args.strata,
                seed=args.seed,
                batch=args.batch,
                device=args.device,
            )
        nll_ar = score_ar(
            ar,
            ids,
            batch=args.batch,
            device=args.device,
            lang_idx=li if ar.cfg.multilingual else None,
        )
        own = nelbo[:, li]
        diff = own - nll_ar
        n = len(ids)
        top1 = nelbo[:, : len(LANG_TO_INDEX)].argmin(axis=1)
        table[lang] = {
            "n_windows": n,
            "n_chars": int(ids.numel()),
            "nelbo_bits": float(own.mean()),
            "nelbo_sem": float(own.std(ddof=1) / np.sqrt(n)),
            "nelbo_uncond_bits": float(nelbo[:, -1].mean()),
            "nll_ar_bits": float(nll_ar.mean()),
            "nll_ar_sem": float(nll_ar.std(ddof=1) / np.sqrt(n)),
            "offset_bits": float(diff.mean()),
            "offset_sem": float(diff.std(ddof=1) / np.sqrt(n)),
            "offset_window_std": float(diff.std(ddof=1)),
            "lid_top1_acc": float((top1 == li).mean()),
            "nelbo_by_condition": {
                name: float(nelbo[:, j].mean()) for name, j in LANG_TO_INDEX.items()
            },
        }
        arrays[f"{lang}/nelbo"] = nelbo
        arrays[f"{lang}/nll_ar"] = nll_ar
        arrays[f"{lang}/doc_index"] = doc_index
        arrays[f"{lang}/doc_ids"] = np.array(heldout.doc_ids[lang])
        t = table[lang]
        print(
            f"  {lang:8s} n={n:4d}  NELBO {t['nelbo_bits']:.4f}±{t['nelbo_sem']:.4f}  "
            f"NLL_AR {t['nll_ar_bits']:.4f}±{t['nll_ar_sem']:.4f}  "
            f"offset {t['offset_bits']:+.4f}±{t['offset_sem']:.4f}  "
            f"LID top-1 {t['lid_top1_acc']:.1%}",
            flush=True,
        )

    offsets = np.array([table[l]["offset_bits"] for l in LANG_TO_INDEX])
    cal_dir = root / "calibration"
    cal_dir.mkdir(exist_ok=True)
    cal = {
        "calibration_version": args.version,
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": args.phase,
        "reference": reference_desc,
        "ar_dir": str(args.ar_dir),
        "definition": "offset_bits = NELBO(own-language condition, EMA backbone) − "
        "NLL_AR, bits/char, mean over the full tiled held-out split; apply as an "
        "additive correction NELBO_calibrated = NELBO − offset in exactly one place "
        "(diff_voyn.metrology.calibration.calibrate_bits, reached through "
        "CalibrationTable.apply / EvaluatorBase.calibrated_bits_per_char).",
        "backbone": bb_meta,
        "nelbo_reused_from": args.nelbo_from,
        "ar_reference": {
            lang: {k: v for k, v in m.items() if k != "config"}
            | {"config": m["config"]}
            for lang, m in ar_meta.items()
        },
        "scoring": {
            "max_windows_cap": args.max_windows,
            "strata": args.strata,
            "seed": args.seed,
            "seq_len": seq_len,
            "crn": "same seed for all conditions within a chunk of --batch windows",
        },
        "offset_spread_bits": float(offsets.max() - offsets.min()),
        "languages": table,
    }
    out = cal_dir / f"calibration_{args.version}.json"
    out.write_text(json.dumps(cal, indent=2))
    np.savez(cal_dir / f"calibration_{args.version}_windows.npz", **arrays)
    print(f"offset spread across languages: {cal['offset_spread_bits']:.4f} bits/char")
    print(f"written: {out}")

    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name=f"calibration-{args.version}", phase="phase3"),
            root,
            tags=["task3.4", "calibration", args.phase],
        )
        task.connect_configuration(cal, name="calibration")
        logger = task.get_logger()
        for lang, t in table.items():
            for key in ("nelbo_bits", "nll_ar_bits", "offset_bits", "lid_top1_acc"):
                logger.report_scalar(f"calibration_{key}", lang, t[key], 0)
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
