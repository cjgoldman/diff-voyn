"""Phase 6, task 6.6 — known-benchmark anchors for the head machinery.

Targets (design §9.5): Zodiac-408 ≤ 1.9% SER, Borg ≤ 4.10%, BnF fr2988
≤ 1.13% — "not because the VMS is these ciphers, but because failing solved
problems invalidates any unsolved-problem claim".

What this instrument can and cannot do with them:

- **Borg** (MSS Borg.lat.898, 17th c., Latin, 34 cipher symbols — the
  rung-2 class at 20–25 letters of plaintext support). Transcription +
  Örneholm's corrected Latin plaintext from matthewdgreen/cipher_benchmark
  (sparse clone, pinned SHA in the report; symbols ``S###``, ``|`` word
  boundaries). It runs through the SAME pipeline as the manuscript
  (``diff_voyn.vms.apply``: rung-2 inner search on a solve window under
  every language hypothesis → paired ELBO + MDL selection → ``elbo_polish``
  → full-stream scoring), so it is both a SER anchor and a language-recovery
  anchor (Latin, with the Italian/Latin near-tie of Phase 3 in play). SER is
  measured page-wise as the Levenshtein error rate between the decode and the
  normalized plaintext on pages whose symbol/letter ratio is within
  ``--align-tol`` of 1 (the published plaintext is corrected and expanded, not
  symbol-aligned; the subset and its size are reported).
- **Zodiac-408** (English, 54 symbols): outside the frozen inventory, so the
  frozen evaluator cannot score it. Reported as the PRE-DIFFUSION n-gram
  baseline only: the rung-2 inner search with an English pentagram trained on
  public-domain modern English (Project Gutenberg, ``data/raw/anchors``,
  sha256 in the report), SER against the canonical plaintext.
- **BnF fr2988**: no transcription is available to us (the benchmark repo
  lists the DECODE/Gallica BnF material as "transcription pending"; DECODE
  itself requires a login) — recorded as NOT RUN.

Stages: prepare → solve (CPU) → score (GPU; Borg only) → report
Artifacts: DATA_ROOT/analysis/phase6/anchors/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "solve" in sys.argv or "prepare" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.ngram import LETTER_TO_IDX, NgramLM, train_lm
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.normalize import normalize
from diff_voyn.vms.apply import (
    KEY,
    cell_from_score,
    instance_record,
    load_instance,
    make_jobs,
    order0_entropy_bits,
    rank_table,
    run_scores,
    run_solves,
)

TARGETS = {"zodiac408": 0.019, "borg": 0.0410, "bnf_fr2988": 0.0113}


def bench_dir(root: Path) -> Path:
    return root / "external" / "cipher_benchmark"


def bench_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=bench_dir(root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _ids(text: str) -> np.ndarray:
    return np.array(
        [LETTER_TO_IDX[c] for c in normalize(text) if c in LETTER_TO_IDX],
        dtype=np.int64,
    )


# -- prepare ---------------------------------------------------------------


def prepare_borg(root: Path, out_dir: Path, min_count: int, align_tol: float) -> dict:
    src = bench_dir(root) / "benchmark" / "sources" / "borg"
    pages = []
    for p in sorted((src / "transcriptions").glob("*.canonical.txt")):
        stem = p.name.replace(".canonical.txt", "")
        pl = src / "plaintext" / f"{stem}.txt"
        if not pl.exists():
            continue
        toks = [t for t in p.read_text().split() if t != "|"]
        plain = _ids(pl.read_text())
        pages.append({"page": stem, "tokens": toks, "plain": plain})
    counts = Counter(t for pg in pages for t in pg["tokens"])
    alphabet = [s for s, c in counts.most_common() if c >= min_count]
    idx = {s: i for i, s in enumerate(alphabet)}
    n_drop = sum(c for s, c in counts.items() if s not in idx)
    stream, page_spans, aligned = [], [], []
    for pg in pages:
        kept = [idx[t] for t in pg["tokens"] if t in idx]
        a = len(stream)
        stream.extend(kept)
        ratio = len(kept) / max(len(pg["plain"]), 1)
        ok = bool(len(kept) >= 50 and abs(ratio - 1.0) <= align_tol)
        page_spans.append(
            {
                "page": pg["page"],
                "span": [a, len(stream)],
                "n_plain": len(pg["plain"]),
                "ratio": ratio,
                "aligned": ok,
            }
        )
        if ok:
            aligned.append(
                {
                    "page": pg["page"],
                    "span": [a, len(stream)],
                    "plain": pg["plain"].tolist(),
                }
            )
    inst = {
        "name": "anchor/borg",
        "kind": "eva",
        "n_symbols": len(alphabet),
        "alphabet": alphabet,
        "n_stream": len(stream),
        "coverage": {
            "n_chars": int(sum(counts.values())),
            "covered_fraction": len(stream) / sum(counts.values()),
            "n_dropped_rare": int(n_drop),
            "n_symbol_types_total": len(counts),
        },
        "symbols": stream,
        "truth": {
            "language": "latin",
            "family": "romance",
            "kind": "borg",
            "aligned_pages": aligned,
        },
        "page_spans": page_spans,
    }
    (out_dir / "borg_eva.json").write_text(json.dumps(inst))
    return {
        k: v for k, v in inst.items() if k not in ("symbols", "truth", "page_spans")
    } | {
        "n_pages": len(pages),
        "n_aligned_pages": len(aligned),
        "aligned_symbols": int(sum(s["span"][1] - s["span"][0] for s in aligned)),
    }


def prepare_zodiac(root: Path, out_dir: Path) -> dict:
    src = bench_dir(root) / "benchmark" / "sources" / "zodiac"
    toks = (
        (src / "transcriptions" / "zodiac408_zenith_global.canonical.txt")
        .read_text()
        .split()
    )
    plain_txt = (src / "plaintext" / "zodiac408_zenith_global.txt").read_text().strip()
    plain = _ids(plain_txt)
    alphabet = sorted(set(toks))
    idx = {s: i for i, s in enumerate(alphabet)}
    stream = [idx[t] for t in toks]
    inst = {
        "name": "anchor/zodiac408",
        "kind": "eva",
        "n_symbols": len(alphabet),
        "alphabet": alphabet,
        "n_stream": len(stream),
        "coverage": {"n_chars": len(stream), "covered_fraction": 1.0},
        "symbols": stream,
        "truth": {
            "language": "english",
            "family": "germanic",
            "kind": "zodiac408",
            "plain_ids": plain.tolist(),
        },
    }
    (out_dir / "zodiac408_eva.json").write_text(json.dumps(inst))
    return {
        "n_symbols": len(alphabet),
        "n_stream": len(stream),
        "n_plain": len(plain),
        "aligned_1to1": len(stream) == len(plain),
    }


def english_lm(root: Path, out_dir: Path, k_max: int = 5) -> tuple[NgramLM, dict]:
    """Witten-Bell pentagram on the Gutenberg texts (held-out: the last one)."""
    files = sorted((root / "raw" / "anchors" / "english").glob("pg*.txt"))
    streams, meta_files = [], []
    for f in files:
        t = f.read_text(encoding="utf-8", errors="replace")
        s = t.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        e = t.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        body = t[t.find("\n", s) + 1 : e] if s >= 0 and e > s else t
        streams.append(_ids(body))
        meta_files.append(
            {
                "file": f.name,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                "n_letters": len(streams[-1]),
            }
        )
    held = streams[-1]
    lm = train_lm("english", streams[:-1], k_max=k_max)
    bits = -lm.score_ids(held, k_max) / np.log(2) / len(held)
    lm.meta.update(
        {"heldout_bits_per_char": float(bits), "files": meta_files, "k_max": k_max}
    )
    info = {
        "files": meta_files,
        "heldout_bits_per_char": float(bits),
        "train_letters": int(sum(len(s) for s in streams[:-1])),
    }
    (out_dir / "english_lm.json").write_text(json.dumps(info, indent=1))
    return lm, info


def stage_prepare(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = data_root()
    info = {
        "benchmark_repo": "https://github.com/matthewdgreen/cipher_benchmark",
        "benchmark_sha": bench_sha(root),
        "borg": prepare_borg(root, args.out_dir, args.min_count, args.align_tol),
        "zodiac408": prepare_zodiac(root, args.out_dir),
        "bnf_fr2988": {
            "status": "NOT RUN",
            "reason": "no transcription available (DECODE login; benchmark repo lists BnF material as transcription pending)",
        },
    }
    _, info["english_lm"] = english_lm(root, args.out_dir)
    (args.out_dir / "prepare.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# -- solve -------------------------------------------------------------------


def stage_solve(args):
    # Borg: the VMS pipeline (rung-2 head, three hypotheses, one solve window)
    rec = instance_record(args.out_dir / "borg_eva.json")
    jobs = make_jobs(
        rec,
        heads=("homophonic",),
        n_windows=args.n_windows,
        w2=args.w2,
        restarts={"homophonic": args.r2},
    )
    run_solves(
        jobs,
        args.out_dir / "borg_solves.json",
        workers=args.workers,
        settings={"w2": args.w2, "r2": args.r2},
        fresh=args.fresh,
    )
    # Zodiac-408: n-gram tier only, English pentagram
    from diff_voyn.heads.evaluator import NgramEvaluator
    from diff_voyn.heads.rung2_homophonic import HomophonicHead

    lm, _ = english_lm(data_root(), args.out_dir)
    ev = NgramEvaluator(
        {"english": lm},
        calibration_offsets_bits={"english": -lm.meta["heldout_bits_per_char"]},
    )
    inst = load_instance(args.out_dir / "zodiac408_eva.json")
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    plain = np.asarray(inst["truth"]["plain_ids"], dtype=np.int64)
    t0 = time.time()
    head = HomophonicHead(ev, seed=0)
    res = head.solve_parallel(
        sym,
        inst["n_symbols"],
        language="english",
        restarts=args.rz,
        workers=args.workers,
        sa_steps=100_000,
        shortlist=12,
    )
    out = []
    for m, s, raw in res.shortlist:
        m2, s2, _ = head.polish_pairs(sym, m, "english")
        best = m2 if s2 > s else m
        out.append(
            {
                "map": best.tolist(),
                "inner": float(max(s, s2)),
                "ser": float(np.mean(best[sym] != plain)),
            }
        )
    rec = {
        "instance": "anchor/zodiac408",
        "language_model": "english (Gutenberg pentagram)",
        "restarts": args.rz,
        "seconds": round(time.time() - t0, 1),
        "shortlist": out,
        "ser_best_by_objective": out[0]["ser"],
        "ser_oracle": min(c["ser"] for c in out),
    }
    write_json_atomic(args.out_dir / "zodiac408_solve.json", rec)
    print(
        "zodiac408: SER by objective",
        rec["ser_best_by_objective"],
        "oracle",
        rec["ser_oracle"],
        f"({rec['seconds']}s)",
    )


# -- score (Borg, GPU) -------------------------------------------------------


def stage_score(args):
    from diff_voyn.heads.diffusion_eval import DiffusionEvaluator

    torch.set_float32_matmul_precision("high")
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
    solves = json.loads((args.out_dir / "borg_solves.json").read_text())["instances"]
    inst = load_instance(args.out_dir / "borg_eva.json")
    i, n = (int(x) for x in args.shard.split("/"))
    run_scores(
        solves,
        {(inst["name"], inst["kind"]): inst},
        args.out_dir
        / (f"borg_scores_shard{i}of{n}.json" if n > 1 else "borg_scores.json"),
        ev=ev,
        budget=args.budget,
        seeds=tuple(range(args.seeds)),
        score_windows=args.score_windows,
        shard=(i, n),
        fresh=args.fresh,
        meta={"evaluator": ev.meta, "budget": args.budget},
    )


# -- report ------------------------------------------------------------------


def _borg_ser(inst: dict, sym_map: np.ndarray) -> dict:
    from diff_voyn.heads.rung4_arithmetic import levenshtein_ser

    sym = np.asarray(inst["symbols"], dtype=np.int64)
    dec = sym_map[sym]
    errs, n = 0.0, 0
    per_page = []
    for pg in inst["truth"]["aligned_pages"]:
        a, b = pg["span"]
        truth = np.asarray(pg["plain"], dtype=np.int64)
        s = levenshtein_ser(dec[a:b], truth)
        per_page.append(s)
        errs += s * len(truth)
        n += len(truth)
    return {
        "ser_weighted": errs / max(n, 1),
        "ser_median_page": float(np.median(per_page)) if per_page else None,
        "n_pages": len(per_page),
        "n_letters": n,
    }


def stage_report(args):
    root = data_root()
    table = CalibrationTable.load(args.primary, root)
    prep = json.loads((args.out_dir / "prepare.json").read_text())
    inst = load_instance(args.out_dir / "borg_eva.json")
    sym = np.asarray(inst["symbols"])
    scores = []
    for sp in sorted(args.out_dir.glob("borg_scores*.json")):
        scores += json.loads(sp.read_text())["instances"]
    scores = list({tuple(r[k] for k in KEY): r for r in scores}.values())
    cells = [
        cell_from_score(
            r,
            table,
            {
                "n_cipher_all": int(inst["coverage"]["n_chars"]),
                "order0_entropy_bits": order0_entropy_bits(sym),
            },
        )
        for r in scores
    ]
    for c, r in zip(cells, scores):
        c["ser"] = _borg_ser(inst, np.asarray(r["final"]["key"]["map"]))
        # n-gram winner's SER (the pre-diffusion baseline) from the solves
    solves = {
        tuple(r[k] for k in KEY): r
        for r in json.loads((args.out_dir / "borg_solves.json").read_text())[
            "instances"
        ]
    }
    for c in cells:
        s = solves[tuple(c[k] for k in KEY)]
        ng = max(s["candidates"], key=lambda x: x["inner"])
        c["ser_ngram_winner"] = _borg_ser(inst, np.asarray(ng["map"]))
        c["ser_oracle_shortlist"] = min(
            _borg_ser(inst, np.asarray(x["map"]))["ser_weighted"]
            for x in s["candidates"]
        )
    tab = rank_table(cells)
    latin = [c for c in cells if c["hypothesis"] == "latin"]
    best_latin = (
        min(latin, key=lambda c: c["total_per_covered_symbol"]) if latin else None
    )
    z = json.loads((args.out_dir / "zodiac408_solve.json").read_text())
    report = {
        "task": "6.6",
        "created_utc": datetime.now(UTC).isoformat(),
        "benchmark_sha": prep["benchmark_sha"],
        "targets": TARGETS,
        "borg": {
            "prepare": prep["borg"],
            "language_rank": tab["per_head"]["homophonic"],
            "abstain": tab["abstain"],
            "cells": cells,
            "ser_final_latin": best_latin["ser"] if best_latin else None,
            "ser_ngram_latin": best_latin["ser_ngram_winner"] if best_latin else None,
            "ser_oracle_latin": (
                best_latin["ser_oracle_shortlist"] if best_latin else None
            ),
            "target": TARGETS["borg"],
            "pass": bool(
                best_latin and best_latin["ser"]["ser_weighted"] <= TARGETS["borg"]
            ),
            "language_recovered": tab["per_head"]["homophonic"]["top"] == "latin",
        },
        "zodiac408": {
            **z,
            "target": TARGETS["zodiac408"],
            "pass": z["ser_best_by_objective"] <= TARGETS["zodiac408"],
            "note": "English is outside the frozen inventory: n-gram tier only (pre-diffusion baseline); no diffusion score",
        },
        "bnf_fr2988": prep["bnf_fr2988"],
    }
    write_json_atomic(args.out_dir / "anchors_report.json", report)
    md = [
        "### Known-benchmark anchors (task 6.6)",
        "",
        "| anchor | class | language | instrument | SER (n-gram winner → final) | target | language recovery | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    b = report["borg"]
    if best_latin:
        md.append(
            f"| Borg (Borg.lat.898) | homophonic, {inst['n_symbols']} symbols ≥{args.min_count} occ. ({inst['coverage']['covered_fraction']:.3f} of symbols) | Latin | full pipeline (rung-2 + ELBO polish, frozen evaluator) | {b['ser_ngram_latin']['ser_weighted']:.4f} → **{b['ser_final_latin']['ser_weighted']:.4f}** (median page {b['ser_final_latin']['ser_median_page']:.4f}; {b['ser_final_latin']['n_pages']} aligned pages, {b['ser_final_latin']['n_letters']} letters; oracle {b['ser_oracle_latin']:.4f}) | ≤ {TARGETS['borg']} | {b['language_rank']['language_order']} (margin {b['language_rank']['margin_top2_bits_per_symbol']:.3f} bits/symbol) | {'PASS' if b['pass'] else 'FAIL'} |"
        )
    zz = report["zodiac408"]
    md.append(
        f"| Zodiac-408 | homophonic, 54 symbols | English (outside inventory) | n-gram tier only, Gutenberg English pentagram ({prep['english_lm']['heldout_bits_per_char']:.2f} bits/char held-out) | {zz['ser_best_by_objective']:.4f} (oracle {zz['ser_oracle']:.4f}, {zz['restarts']} restarts) | ≤ {TARGETS['zodiac408']} | n/a | {'PASS' if zz['pass'] else 'FAIL'} (pre-diffusion baseline) |"
    )
    md.append(
        f"| BnF fr2988 | homophonic/nomenclator | French (outside inventory) | — | — | ≤ {TARGETS['bnf_fr2988']} | — | NOT RUN ({prep['bnf_fr2988']['reason']}) |"
    )
    md.append("")
    for c in sorted(cells, key=lambda c: c["total_per_covered_symbol"]):
        md.append(
            f"- Borg hyp={c['hypothesis']} w{c['window']}: plain {c['plain_bits']:.3f} ± {c['plain_bits_sem']:.3f} bits/char, total {c['total_per_covered_symbol']:.3f} bits/symbol, structure margin {c['structure_margin']:.2f}, SER {c['ser']['ser_weighted']:.4f} (n-gram {c['ser_ngram_winner']['ser_weighted']:.4f}), language-like {c['language_like']}, polish accepted {c['elbo_polish_accepted']}"
        )
    md = "\n".join(md)
    (args.out_dir / "anchors_report.md").write_text(md)
    print(md)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = data_root()
    p.add_argument(
        "--stage", choices=["prepare", "solve", "score", "report"], required=True
    )
    p.add_argument(
        "--out-dir", type=Path, default=root / "analysis" / "phase6" / "anchors"
    )
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--align-tol", type=float, default=0.08)
    p.add_argument("--w2", type=int, default=4000)
    p.add_argument("--n-windows", type=int, default=2)
    p.add_argument("--r2", type=int, default=96)
    p.add_argument("--rz", type=int, default=240, help="Zodiac-408 SA restarts")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--score-windows", type=int, default=16)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--primary", default=CALIBRATION_VERSION)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {
        "prepare": stage_prepare,
        "solve": stage_solve,
        "score": stage_score,
        "report": stage_report,
    }[args.stage](args)


if __name__ == "__main__":
    main()
