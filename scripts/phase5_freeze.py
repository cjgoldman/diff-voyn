"""Task 5.1 — freeze the Phase-5 evaluator and verify the head interface
against the REAL weights.

Design §7.4: the backbone is frozen (EMA weights) during all cipher-head
optimization. This script (a) names and fingerprints the frozen evaluator
(the Gate-G4 joint Phase-C 85M EMA checkpoint, adopted calibration table),
(b) re-runs the CH.4 smoke tests against it — gradients reach a toy head's
parameters, the ``logaddexp(−∞,−∞)`` NULL-blend corners are finite, the
one-hot frame path equals the id path — and (c) checks that the soft frame
path and the Phase-3 metrology estimator agree exactly on one-hot input at
the same seed (the frame path is the same instrument as the G3/G4 numbers),
plus measures the 2N-NULL-frame vs plain-stream bits gap and the cost of a
scoring call. Nothing here changes any weight.

Usage:
    uv run python scripts/phase5_freeze.py [--ckpt ...] [--device cuda]
Writes DATA_ROOT/analysis/phase5/evaluator_freeze.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.data.noise import P_UNIGRAM_NAIBBE, frame_with_nulls
from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
from diff_voyn.heads.frame import build_frame, letters_to_vocab
from diff_voyn.heads.ngram import A
from diff_voyn.heads.synth import HeldoutSampler
from diff_voyn.heads.two_tier import paired_bits
from diff_voyn.infra.nelbo import per_window_nelbo_bits
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable
from diff_voyn.vocab import LETTER_IDS, NULL_ID, VOCAB_SIZE

LANGS = tuple(LANG_TO_INDEX)
CHECKS: list[dict] = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sha256(path: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument(
        "--sibling", type=Path, default=root / "runs/phase_c-25m-seed0/ckpt_final.pt"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out_dir = root / "analysis" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    t0 = time.time()
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device, n_strata=8)
    table = CalibrationTable.load(CALIBRATION_VERSION)
    print(f"loaded {args.ckpt} in {time.time()-t0:.1f}s: {ev.meta}")
    freeze = {
        "evaluator": {
            **ev.meta,
            "sha256": sha256(args.ckpt),
            "n_params": int(sum(p.numel() for p in ev.backbone.parameters())),
        },
        "sibling_25m": (
            {"path": str(args.sibling), "sha256": sha256(args.sibling)}
            if args.sibling.exists()
            else None
        ),
        "calibration": table.summary(),
        "additive_offsets_bits": ev.calibration_offsets_bits,
    }
    check(
        "5.1 evaluator frozen: EMA weights, no parameter requires grad, eval mode",
        ev.meta["weights"] == "ema"
        and not any(q.requires_grad for q in ev.backbone.parameters())
        and not ev.backbone.training,
        f"{args.ckpt.name} step {ev.meta['step']} ({ev.meta['weights']}), calibration {CALIBRATION_VERSION} ({table.policy})",
    )

    # --- one-hot frame path == id path -----------------------------------
    ids = torch.randint(LETTER_IDS[0], LETTER_IDS[-1] + 1, (1, 64), device=ev.device)
    onehot = torch.nn.functional.one_hot(ids, VOCAB_SIZE).float()
    lang = torch.tensor([0], device=ev.device)
    with torch.no_grad():
        a = ev.backbone(ids, lang).float()
        b = ev.backbone.forward_soft(onehot, lang).float()
    # logits are finite except the SUBS -inf MASK column
    fin = torch.isfinite(a)
    diff = (a[fin] - b[fin]).abs().max().item()
    check(
        "5.1 forward_soft on one-hot == id forward",
        diff < 1e-3,
        f"max |Δlogit| {diff:.2e}",
    )

    # --- gradients reach a toy head through the frozen backbone ------------
    torch.manual_seed(args.seed)
    head_logits = torch.randn(2 * 24, A, requires_grad=True)
    w_logit = torch.zeros(24, requires_grad=True)
    soft = torch.softmax(head_logits, 1)
    frame = build_frame(soft[0::2], soft[1::2], torch.sigmoid(w_logit))
    score = ev.score_frame(frame, language="latin")
    score.backward()
    ok = (
        torch.isfinite(score).item()
        and torch.isfinite(head_logits.grad).all().item()
        and torch.isfinite(w_logit.grad).all().item()
        and float(head_logits.grad.abs().max()) > 0
        and all(q.grad is None for q in ev.backbone.parameters())
    )
    check(
        "5.1 gradients reach toy-head parameters; backbone grads absent",
        ok,
        f"score {float(score):.2f} nats; |grad| max head {float(head_logits.grad.abs().max()):.3e}, w {float(w_logit.grad.abs().max()):.3e}",
    )

    # --- NULL-blend corners (w = 0 / 1 exactly, the logaddexp trap) --------
    corners_ok = True
    details = []
    for w in (0.0, 1.0, 0.5):
        hl = torch.randn(2 * 16, A, requires_grad=True)
        s = torch.softmax(hl, 1)
        fr = build_frame(s[0::2], s[1::2], torch.full((16,), w))
        sc = ev.score_frame(fr, language="italian")
        sc.backward()
        fin = bool(torch.isfinite(sc)) and bool(torch.isfinite(hl.grad).all())
        corners_ok &= fin
        details.append(f"w={w}: score {float(sc):.1f} grad-finite {fin}")
    # hard one-hot letters with a hard NULL slot (frame exactly 0/1)
    hard = torch.zeros(8, VOCAB_SIZE)
    hard[0::2, LETTER_IDS[0]] = 1.0
    hard[1::2, NULL_ID] = 1.0
    hard.requires_grad_(True)
    sc = ev.score_frame(hard, language="german")
    sc.backward()
    corners_ok &= bool(torch.isfinite(sc)) and bool(torch.isfinite(hard.grad).all())
    details.append(f"hard 0/1 frame: finite {bool(torch.isfinite(sc))}")
    check(
        "5.1 logaddexp(-inf,-inf) smoke test (NULL-blend corners)",
        corners_ok,
        "; ".join(details),
    )

    # --- CRN determinism ---------------------------------------------------
    with torch.no_grad():
        fr = frame.detach()
        s1 = float(ev.score_frame(fr, language="latin"))
        s2 = float(ev.score_frame(fr, language="latin"))
        s3 = float(ev.score_frame(fr, language="latin", seed=1))
        s4 = float(ev.score_frame(fr, language="italian"))
    check(
        "5.1 masking is a pure function of the seed (CRN)",
        s1 == s2 and s1 != s3 and s1 != s4,
        f"same seed {s1:.4f} == {s2:.4f}; other seed {s3:.4f}; other language {s4:.4f}",
    )

    # --- soft frame path == metrology estimator on one-hot input -----------
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    rng = np.random.default_rng(args.seed)
    agree = []
    gap_rows = []
    timing = {}
    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        letters = sampler.sample(700, rng)
        ids_t = torch.from_numpy(letters + LETTER_IDS[0])[None]
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=ev.autocast),
        ):
            metro = per_window_nelbo_bits(
                ev.backbone,
                ids_t,
                LANG_TO_INDEX[lang],
                n_strata=16,
                seed=123,
                device=ev.device,
            )[0].item()
        frame1 = letters_to_vocab(
            torch.nn.functional.one_hot(torch.from_numpy(letters), A).float()
        )
        with torch.no_grad():
            soft_nats = float(
                ev.score_frame(frame1, language=lang, seed=123, n_strata=16)
            )
        soft_bits = -soft_nats / (len(letters) * math.log(2.0))
        pb = paired_bits(ev, letters[None], [lang], n_strata=16, seed=123)[0, 0]
        hard_bits = ev.score_ids(letters[None], language=lang, n_strata=16, seed=123)[
            0, 0
        ]
        agree.append((lang, metro, soft_bits, pb, hard_bits))
        # 2N NULL frame vs plain stream (Phase-B frame layout)
        framed, finfo = frame_with_nulls(letters + LETTER_IDS[0], rng, P_UNIGRAM_NAIBBE)
        fr2 = torch.nn.functional.one_hot(
            torch.from_numpy(framed.astype(np.int64)), VOCAB_SIZE
        ).float()
        with torch.no_grad():
            all_nats = float(ev.score_frame(fr2, language=lang, seed=123, n_strata=64))
            let_nats = float(
                ev.score_frame(
                    fr2, language=lang, seed=123, n_strata=64, letter_slots_only=True
                )
            )
            plain_bits = ev.score_ids(
                letters[None], language=lang, n_strata=64, seed=123
            )[0, 0]
        gap_rows.append(
            {
                "language": lang,
                "plain_bits_per_char": float(plain_bits),
                "frame_letter_slots_bits_per_plain_char": -let_nats
                / (len(letters) * math.log(2.0)),
                "frame_all_slots_bits_per_plain_char": -all_nats
                / (len(letters) * math.log(2.0)),
                "frame_null_fraction": finfo["null_fraction"],
                "n_slots": int(fr2.shape[0]),
            }
        )
    worst = max(abs(m - s) for _, m, s, _, _ in agree)
    worst_pb = max(abs(m - q) for _, m, _, q, _ in agree)
    worst_hard = max(abs(m - h) for _, m, _, _, h in agree)
    check(
        "5.1 soft frame path reproduces the Phase-3 metrology estimator on one-hot input (same seed)",
        worst < 2e-2 and worst_pb < 1e-3 and worst_hard < 1e-3,
        "; ".join(
            f"{l}: metrology {m:.4f} / frame {s:.4f} / paired {q:.4f} / score_ids {h:.4f}"
            for l, m, s, q, h in agree
        ),
    )
    for r in gap_rows:
        print(
            f"  NULL-frame gap {r['language']}: plain {r['plain_bits_per_char']:.4f}  "
            f"frame letter-slots {r['frame_letter_slots_bits_per_plain_char']:.4f}  "
            f"frame all-slots {r['frame_all_slots_bits_per_plain_char']:.4f} bits/plain char"
        )

    # --- cost --------------------------------------------------------------
    if ev.device.type == "cuda":
        torch.cuda.synchronize()
    xb = torch.randint(LETTER_IDS[0], LETTER_IDS[-1] + 1, (16, 1024), device=ev.device)
    lb = torch.zeros(16, dtype=torch.long, device=ev.device)
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=ev.autocast),
    ):
        ev.backbone(xb, lb)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(5):
            ev.backbone(xb, lb)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
    timing["forward_16x1024_s"] = (time.time() - t1) / 5
    hl = torch.randn(700, A, requires_grad=True)
    t1 = time.time()
    for _ in range(3):
        sc = ev.score_fixed(torch.softmax(hl, 1), language="latin")
        sc.backward()
    if ev.device.type == "cuda":
        torch.cuda.synchronize()
    timing["grad_step_700_slots_8_strata_s"] = (time.time() - t1) / 3
    t1 = time.time()
    paired_bits(ev, np.stack([letters] * 8), list(LANGS), n_strata=64, seed=0)
    timing["paired_8cands_x3cond_700_b64_s"] = time.time() - t1
    print("timing:", json.dumps(timing, indent=1))

    verdict = all(c["status"] == "PASS" for c in CHECKS)
    report = {
        "task": "5.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if verdict else "FAIL",
        "checks": CHECKS,
        "frozen": freeze,
        "estimator_agreement": [
            {"language": l, "metrology": m, "frame": s, "paired": q, "score_ids": h}
            for l, m, s, q, h in agree
        ],
        "null_frame_gap": gap_rows,
        "timing": timing,
        "device": str(ev.device),
    }
    out = out_dir / "evaluator_freeze.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nTask 5.1: {report['verdict']}  (report {out})")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
