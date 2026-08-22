"""Task 4.6 — temperature-scale the language-ID head on held-out
decipherments and report the ELBO-ranking vs head-ranking agreement.

Calibration set: the rung-1 decipherments of the 3.6 suite
(``recovery_solves.json``; held-out plaintext, every cipher's *true*
hypothesis decipherment, label = its language) plus one shuffled version of
each plaintext (label = abstain). Trials are split by parity: even trials
fit the temperature (one scalar, NLL-minimized), odd trials report NLL /
ECE / accuracy before and after. The fitted temperature is written into a
standalone head checkpoint next to the joint one
(``<run_dir>/lid_head_calibrated.pt``; ``lid_eval.py --calibrated`` and the
Phase-6 consumers load that); the joint checkpoint is never edited.

Agreement diagnostic (design §6: "its disagreement with the ELBO ranking is
itself a diagnostic worth reporting"): for every instance the head's top
class (languages + abstain) is crossed with the ELBO ranking's winner under
the adopted calibration table (``recovery_scores.json`` of the same
backbone), per length and overall, with the agreement rate, the
head-abstains rate and the rate both are right.

Usage:
    uv run python scripts/head_calibration.py \\
        --ckpt DATA_ROOT/runs/phase_c-85m-seed0/ckpt_final.pt \\
        --scores DATA_ROOT/analysis/phase4/recovery_scores.json
Writes DATA_ROOT/analysis/phase4/head_calibration.{json,md}; ClearML tag
``task4.6``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from diff_voyn.ciphers.external import data_root
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.data.noise import LETTER_BASE
from diff_voyn.infra.checkpoint import load_backbone, load_lid_head, save_checkpoint
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable, rank_languages
from diff_voyn.model.lid_head import (
    ABSTAIN_CLASS,
    LID_CLASSES,
    lid_logits,
)

LANGS = tuple(LANG_TO_INDEX)


@torch.no_grad()
def raw_logits(backbone, head, ids: np.ndarray, device, batch=32) -> torch.Tensor:
    backbone.eval()
    head.eval()
    out = []
    x = torch.from_numpy(np.asarray(ids).astype(np.int64))
    for s in range(0, len(x), batch):
        chunk = x[s : s + batch].to(device)
        g = torch.Generator().manual_seed(s)
        out.append(
            lid_logits(backbone, head, chunk, g=g, stop_gradient=True, autocast=True)
            .float()
            .cpu()
        )
    return torch.cat(out)


def nll(logits: torch.Tensor, labels: torch.Tensor, T: float) -> float:
    return float(F.cross_entropy(logits / T, labels))


def ece(logits: torch.Tensor, labels: torch.Tensor, T: float, bins: int = 15) -> float:
    p = F.softmax(logits / T, dim=-1)
    conf, pred = p.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            total += float(
                m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
            )
    return total


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    grid = np.exp(np.linspace(np.log(0.2), np.log(10.0), 241))
    best = min(grid, key=lambda T: nll(logits, labels, float(T)))
    lo, hi = best / 1.1, best * 1.1  # golden-section refinement
    phi = (5**0.5 - 1) / 2
    for _ in range(40):
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        if nll(logits, labels, float(a)) < nll(logits, labels, float(b)):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument("--ckpt", type=Path, required=True, help="joint Phase-C checkpoint")
    p.add_argument("--head", type=Path, default=None)
    p.add_argument(
        "--solves", type=Path, default=root / "analysis/phase3/recovery_solves.json"
    )
    p.add_argument(
        "--scores", type=Path, default=root / "analysis/phase4/recovery_scores.json"
    )
    p.add_argument("--calibration", default=CALIBRATION_VERSION)
    p.add_argument("--out-dir", type=Path, default=root / "analysis/phase4")
    p.add_argument("--tag", default="")
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    backbone, bmeta = load_backbone(args.ckpt, device)
    head, hmeta = load_lid_head(args.head or args.ckpt, device)
    table = CalibrationTable.load(args.calibration, root)
    offs = table.additive_offsets()
    print(
        f"backbone {bmeta['run_name']} step {bmeta['step']}; head step {hmeta['step']} "
        f"({hmeta['weights']}); ELBO ranking under {table.version} ({table.policy})"
    )

    solves = json.loads(args.solves.read_text())["instances"]
    scores = {
        (r["language"], r["length"], r["trial"]): r
        for r in json.loads(args.scores.read_text())["instances"]
    }
    t0 = time.time()
    rows = []
    rng = np.random.default_rng(46)
    # group by length so every forward batch has one width
    by_len: dict[int, list] = {}
    for r in solves:
        by_len.setdefault(r["length"], []).append(r)
    for L, rs in sorted(by_len.items()):
        true_dec = (
            np.array([r["decipherments"][r["language"]] for r in rs]) + LETTER_BASE
        )
        plain = np.array([r["plain_ids"] for r in rs]) + LETTER_BASE
        shuf = np.stack([rng.permutation(row) for row in plain])
        lg_true = raw_logits(backbone, head, true_dec, device)
        lg_shuf = raw_logits(backbone, head, shuf, device)
        for i, r in enumerate(rs):
            key = (r["language"], r["length"], r["trial"])
            elbo_winner = None
            if key in scores:
                mean_bits = {
                    h: float(np.mean(v))
                    for h, v in scores[key]["diffusion_bits"].items()
                }
                elbo_winner = rank_languages(mean_bits, offs)[0][0]
            rows.append(
                {
                    "language": r["language"],
                    "length": L,
                    "trial": r["trial"],
                    "kind": "true_decipherment",
                    "label": LANG_TO_INDEX[r["language"]],
                    "logits": lg_true[i],
                    "elbo_winner": elbo_winner,
                    "ser": r["ser_true_hypothesis"],
                }
            )
            rows.append(
                {
                    "language": r["language"],
                    "length": L,
                    "trial": r["trial"],
                    "kind": "shuffled",
                    "label": ABSTAIN_CLASS,
                    "logits": lg_shuf[i],
                    "elbo_winner": None,
                    "ser": None,
                }
            )
        print(f"  L={L}: {len(rs)} instances ({time.time()-t0:.0f}s)", flush=True)

    logits = torch.stack([r["logits"] for r in rows])
    labels = torch.tensor([r["label"] for r in rows])
    fit = torch.tensor([r["trial"] % 2 == 0 for r in rows])
    T = fit_temperature(logits[fit], labels[fit])
    test = ~fit

    def metrics(mask, Tv):
        lg, lb = logits[mask], labels[mask]
        return {
            "n": int(mask.sum()),
            "nll": nll(lg, lb, Tv),
            "ece": ece(lg, lb, Tv),
            "acc": float(((lg / Tv).argmax(-1) == lb).float().mean()),
            "mean_confidence": float(F.softmax(lg / Tv, -1).max(-1).values.mean()),
        }

    calib = {
        "temperature": T,
        "fit_split": "even trials",
        "test_split": "odd trials",
        "before": {"fit": metrics(fit, 1.0), "test": metrics(test, 1.0)},
        "after": {"fit": metrics(fit, T), "test": metrics(test, T)},
        "by_length_test": {},
    }
    for L in sorted(by_len):
        m = test & torch.tensor([r["length"] == L for r in rows])
        calib["by_length_test"][str(L)] = {
            "before": metrics(m, 1.0),
            "after": metrics(m, T),
        }

    # agreement matrix (true decipherments only, calibrated head)
    probs = F.softmax(logits / T, -1)
    pred = probs.argmax(-1)
    agreement: dict = {}
    overall = {
        "n": 0,
        "agree": 0,
        "head_abstain": 0,
        "head_true": 0,
        "elbo_true": 0,
        "both_true": 0,
    }
    matrix_all = np.zeros((len(LID_CLASSES), len(LANGS)), dtype=int)
    for L in sorted(by_len):
        matrix = np.zeros((len(LID_CLASSES), len(LANGS)), dtype=int)
        cnt = {
            "n": 0,
            "agree": 0,
            "head_abstain": 0,
            "head_true": 0,
            "elbo_true": 0,
            "both_true": 0,
        }
        for i, r in enumerate(rows):
            if (
                r["kind"] != "true_decipherment"
                or r["length"] != L
                or r["elbo_winner"] is None
            ):
                continue
            hp = int(pred[i])
            ew = LANG_TO_INDEX[r["elbo_winner"]]
            matrix[hp, ew] += 1
            cnt["n"] += 1
            cnt["agree"] += int(hp == ew)
            cnt["head_abstain"] += int(hp == ABSTAIN_CLASS)
            cnt["head_true"] += int(hp == r["label"])
            cnt["elbo_true"] += int(ew == r["label"])
            cnt["both_true"] += int(hp == r["label"] and ew == r["label"])
        matrix_all += matrix
        for k in overall:
            overall[k] += cnt[k]
        agreement[str(L)] = {
            **{k: (v / cnt["n"] if k != "n" else v) for k, v in cnt.items()},
            "matrix_head_rows_elbo_cols": matrix.tolist(),
        }
    agreement["overall"] = {
        **{k: (v / overall["n"] if k != "n" else v) for k, v in overall.items()},
        "matrix_head_rows_elbo_cols": matrix_all.tolist(),
    }
    long_keys = [str(L) for L in sorted(by_len) if L >= 200]
    agree_ge200 = (
        float(np.mean([agreement[k]["agree"] for k in long_keys]))
        if long_keys
        else None
    )

    # calibrated head checkpoint
    head.log_temperature.fill_(float(np.log(T)))
    run_dir = args.ckpt.parent
    out_ckpt = run_dir / f"lid_head_calibrated{args.tag}.pt"
    save_checkpoint(
        out_ckpt,
        model=head,
        step=hmeta["step"],
        extra={
            "lid_head_config": hmeta["lid_head_config"],
            "backbone": bmeta,
            "config": {"phase": "phase_c_calibrated", "run_name": bmeta["run_name"]},
            "temperature": T,
            "calibration_source": str(args.solves),
        },
    )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "task": "4.6",
        "backbone": bmeta,
        "head": hmeta,
        "elbo_ranking_table": table.summary(),
        "calibration": calib,
        "agreement": agreement,
        "agreement_ge200_mean": agree_ge200,
        "calibrated_head_checkpoint": str(out_ckpt),
        "rows": [
            {k: v for k, v in r.items() if k != "logits"}
            | {"head_probs": probs[i].tolist(), "head_pred": LID_CLASSES[int(pred[i])]}
            for i, r in enumerate(rows)
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"head_calibration{args.tag}.json"
    out.write_text(json.dumps(report, indent=1))
    md = render_markdown(report)
    (args.out_dir / f"head_calibration{args.tag}.md").write_text(md)
    print(md)
    print(f"written {out}; calibrated head {out_ckpt}")
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name=f"head-calibration{args.tag}", phase="phase4"),
            root,
            tags=["task4.6"],
        )
        task.connect_configuration(
            {k: v for k, v in report.items() if k != "rows"}, name="head_calibration"
        )
        logger = task.get_logger()
        for L, a in agreement.items():
            if L != "overall":
                logger.report_scalar("head_vs_elbo", "agreement", a["agree"], int(L))
                logger.report_scalar(
                    "head_vs_elbo", "head_true", a["head_true"], int(L)
                )
                logger.report_scalar(
                    "head_vs_elbo", "elbo_true", a["elbo_true"], int(L)
                )
        logger.flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()


def render_markdown(rep: dict) -> str:
    c = rep["calibration"]
    lines = [
        f"### Head calibration (task 4.6) — backbone step {rep['backbone']['step']}, T = {c['temperature']:.3f}",
        "",
        "| split | NLL before → after | ECE before → after | acc | mean conf. after |",
        "|---|---|---|---|---|",
    ]
    for sp in ("fit", "test"):
        b, a = c["before"][sp], c["after"][sp]
        lines.append(
            f"| {sp} (n={a['n']}) | {b['nll']:.4f} → {a['nll']:.4f} | {b['ece']:.4f} → {a['ece']:.4f} | "
            f"{a['acc']:.3f} | {a['mean_confidence']:.3f} |"
        )
    lines += ["", "Test split by length (ECE before → after, accuracy):", ""]
    for L, d in c["by_length_test"].items():
        lines.append(
            f"- L{L}: {d['before']['ece']:.4f} → {d['after']['ece']:.4f}, acc {d['after']['acc']:.3f}"
        )
    lines += [
        "",
        f"**Head vs ELBO ranking** (true-hypothesis decipherments; ELBO under `{rep['elbo_ranking_table']['version']}`):",
        "",
        "| L | n | agree | head abstains | head right | ELBO right | both right |",
        "|---|---|---|---|---|---|---|",
    ]
    for L, a in rep["agreement"].items():
        lines.append(
            f"| {L} | {a['n']} | {a['agree']:.3f} | {a['head_abstain']:.3f} | {a['head_true']:.3f} | "
            f"{a['elbo_true']:.3f} | {a['both_true']:.3f} |"
        )
    m = rep["agreement"]["overall"]["matrix_head_rows_elbo_cols"]
    lines += [
        "",
        "Overall matrix (rows = head class, columns = ELBO winner "
        + "/".join(LANGS)
        + "):",
        "",
    ]
    for cls, row in zip(LID_CLASSES, m):
        lines.append(f"- {cls}: {row}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
