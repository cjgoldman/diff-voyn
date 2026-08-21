"""Task 3.3 — per-window document scoring with mean and spread.

Every held-out document, cut into consecutive 1024-char windows, scored
under every conditioning language (+ unconditional) with CRN; reported per
document as mean ± window std / s.e.m., with the calibrated top-1 language
per document and per window, so rankings carry uncertainty — and so the
heterogeneity of a held-out set (the Latin split spans 2.1–3.0 bits/char by
document, `docs/phase1_status.md`) is visible instead of averaged away.

By default reuses the per-window arrays of a calibration run
(``--from-calibration <version>``, same windows/seeds — no GPU needed);
``--ckpt`` scores fresh with the harness instead.

Usage:
    uv run python scripts/score_documents.py --from-calibration v2
    uv run python scripts/score_documents.py --ckpt DATA_ROOT/runs/.../ckpt_final.pt
Writes DATA_ROOT/analysis/phase3/documents_<tag>.{json,md}; ClearML ``task3.3``.
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
from diff_voyn.metrology import (
    CalibrationTable,
    ScoreSettings,
    family_of,
    per_document,
    rank_languages,
    score_conditions,
)
from diff_voyn.metrology.scoring import DEFAULT_CONDITIONS, spread_summary

LANGS = tuple(LANG_TO_INDEX)


def load_from_calibration(root: Path, version: str):
    cal = json.loads(CalibrationTable.file_for(version, root).read_text())
    npz = np.load(root / "calibration" / f"calibration_{version}_windows.npz")
    per_lang = {}
    heldout = None
    for lang in LANGS:
        if f"{lang}/doc_index" in npz:
            doc_index = npz[f"{lang}/doc_index"]
            doc_ids = [str(d) for d in npz[f"{lang}/doc_ids"]]
        else:  # older table: tiling is deterministic, rebuild the index
            if heldout is None:
                corpus_dir = root / "corpora" / "v1"
                splits = load_splits(corpus_dir)
                heldout = CorpusWindows(
                    corpus_dir,
                    {l: [d["doc_id"] for d in sp["heldout"]]
                     for l, sp in splits["languages"].items()},
                )
            seq_len = cal["backbone"]["model"]["seq_len"]
            tiled, doc_index = heldout.tiled_windows_by_doc(lang, seq_len)
            doc_ids = heldout.doc_ids[lang]
            if len(tiled) != len(npz[f"{lang}/nelbo"]):
                raise SystemExit(f"{lang}: window count mismatch rebuilding doc index")
        per_lang[lang] = (npz[f"{lang}/nelbo"], doc_index, doc_ids)
    meta = {
        "source": f"calibration_{version}",
        "backbone": cal["backbone"],
        "scoring": cal["scoring"],
    }
    return per_lang, meta


def score_fresh(
    root: Path, ckpt: Path, strata: int, seed: int, batch: int, device: str
):
    from diff_voyn.infra.checkpoint import load_backbone

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    heldout = CorpusWindows(
        corpus_dir,
        {
            l: [d["doc_id"] for d in sp["heldout"]]
            for l, sp in splits["languages"].items()
        },
    )
    model, bb = load_backbone(ckpt, device)
    st = ScoreSettings(n_strata=strata, seed=seed, batch=batch)
    per_lang = {}
    for lang in LANGS:
        tiled, doc_index = heldout.tiled_windows_by_doc(lang, bb["model"]["seq_len"])
        scores = score_conditions(
            model, tiled, DEFAULT_CONDITIONS, settings=st, device=device
        )
        per_lang[lang] = (scores, doc_index, heldout.doc_ids[lang])
        print(f"  scored {lang}: {len(tiled)} windows", flush=True)
    return per_lang, {"source": str(ckpt), "backbone": bb, "scoring": st.as_dict()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument("--from-calibration", default=None)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--calibration", default=None, help="table used for the ranking")
    p.add_argument("--tag", default=None)
    p.add_argument("--strata", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not (args.from_calibration or args.ckpt):
        p.error("give --from-calibration or --ckpt")
    if args.from_calibration:
        per_lang, meta = load_from_calibration(root, args.from_calibration)
        tag = args.tag or args.from_calibration
    else:
        if args.device == "cuda":
            torch.set_float32_matmul_precision("high")
        per_lang, meta = score_fresh(
            root, args.ckpt, args.strata, args.seed, args.batch, args.device
        )
        tag = args.tag or args.ckpt.parent.name
    table = CalibrationTable.load(
        args.calibration or args.from_calibration or None, root
    )
    offs = table.additive_offsets()

    conds = list(DEFAULT_CONDITIONS)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "3.3",
        "meta": meta,
        "calibration": table.summary(),
        "languages": {},
    }
    for lang in LANGS:
        scores, doc_index, doc_ids = per_lang[lang]
        docs = per_document(scores, doc_index, doc_ids, conds)
        own = scores[:, LANG_TO_INDEX[lang]]
        win_top1 = [
            rank_languages({h: scores[i, LANG_TO_INDEX[h]] for h in LANGS}, offs)[0][0]
            for i in range(len(scores))
        ]
        entries = []
        for d in docs:
            top1 = rank_languages({h: d.mean[h] for h in LANGS}, offs)[0][0]
            rows = d.window_rows
            entries.append(
                d.as_dict()
                | {
                    "calibrated_own_bits": table.apply(d.mean[lang], lang),
                    "doc_top1": top1,
                    "doc_top1_correct": top1 == lang,
                    "doc_family_correct": family_of(top1) == family_of(lang),
                    "window_top1_acc": float(
                        np.mean([win_top1[r] == lang for r in rows])
                    ),
                    "window_family_acc": float(
                        np.mean(
                            [family_of(win_top1[r]) == family_of(lang) for r in rows]
                        )
                    ),
                }
            )
        doc_means = np.array([d.mean[lang] for d in docs])
        report["languages"][lang] = {
            "n_documents": len(docs),
            "n_windows": len(scores),
            "own_condition_windows": spread_summary(own),
            "own_condition_document_means": spread_summary(doc_means),
            "between_document_std": (
                float(doc_means.std(ddof=1)) if len(docs) > 1 else 0.0
            ),
            "within_document_std_mean": float(
                np.mean([d.std[lang] for d in docs if d.n_windows > 1])
            ),
            "document_top1_acc": float(
                np.mean([e["doc_top1_correct"] for e in entries])
            ),
            "window_top1_acc": float(np.mean([w == lang for w in win_top1])),
            "window_family_acc": float(
                np.mean([family_of(w) == family_of(lang) for w in win_top1])
            ),
            "documents": sorted(entries, key=lambda e: e["mean"][lang]),
        }
    out_dir = root / "analysis" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"documents_{tag}.json"
    out.write_text(json.dumps(report, indent=1))
    md = render(report)
    (out_dir / f"documents_{tag}.md").write_text(md)
    print(md)
    print(f"written {out}")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name=f"documents-{tag}", phase="phase3"),
            root,
            tags=["task3.3"],
        )
        task.connect_configuration(report, name="documents")
        logger = task.get_logger()
        for lang, r in report["languages"].items():
            for e in r["documents"]:
                logger.report_scalar("document_own_bits", lang, e["mean"][lang], 0)
            logger.report_scalar("document_top1_acc", lang, r["document_top1_acc"], 0)
            logger.report_scalar("window_top1_acc", lang, r["window_top1_acc"], 0)
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


def render(rep: dict) -> str:
    lines = [
        f"### Per-document scoring ({rep['meta']['source']}, calibration {rep['calibration']['version']})",
        "",
        "| language | document | windows | own bits/char mean ± std (s.e.m.) | calibrated | doc top-1 | window top-1 acc | window family acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for lang, r in rep["languages"].items():
        for e in r["documents"]:
            name = e["doc_id"][:48]
            lines.append(
                f"| {lang} | `{name}` | {e['n_windows']} | {e['mean'][lang]:.3f} ± {e['std'][lang]:.3f} "
                f"({e['sem'][lang]:.3f}) | {e['calibrated_own_bits']:.3f} | {e['doc_top1']} | "
                f"{e['window_top1_acc']:.1%} | {e['window_family_acc']:.1%} |"
            )
        s = r["own_condition_windows"]
        lines.append(
            f"| **{lang}** | *all ({r['n_documents']} docs)* | {r['n_windows']} | "
            f"{s['mean']:.3f} ± {s['std']:.3f} ({s['sem']:.3f}); between-doc std "
            f"{r['between_document_std']:.3f}, within-doc {r['within_document_std_mean']:.3f} | | "
            f"{r['document_top1_acc']:.0%} docs | {r['window_top1_acc']:.1%} | {r['window_family_acc']:.1%} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
