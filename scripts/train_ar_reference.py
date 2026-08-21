"""Train the per-language char-AR reference models — task 3.4 (design §5b.3).

One small causal transformer per frozen language (``CharARLM``, ~10M params),
trained on exactly the backbone's train split for that language, identical
budget and hyper-parameters for every language. Held-out NLL is tracked on
the full tiled held-out set every ``--eval-every`` steps; both the final and
the best-held-out checkpoint are kept (the calibration script defaults to
the best one — the standard way to report a small LM's cross-entropy; the
final-vs-best gap is recorded so any overfitting is visible, not hidden).

Multilingual reference (``--multilingual``, calibration v3): ONE
language-conditioned model (``ARConfig.multilingual``), trained on the same
τ=0.7-balanced three-language mix as the backbone (``LanguageSampler`` over
the same train split), with the backbone's conditioning dropout. Held-out NLL
is tracked per language; "best" is the checkpoint with the lowest
equal-weight mean over languages (one selection for all languages — no
per-language cherry-picking, which would itself be a calibration bias).

Usage:
    uv run python scripts/train_ar_reference.py [--lang latin ...] [--steps N]
    uv run python scripts/train_ar_reference.py --multilingual --preset 25m \
        --version v3 --steps 20000

Artifacts: DATA_ROOT/ar_reference/<version>/<lang>/{ar_best.pt, ar_final.pt}
(or ``<version>/multilingual/`` for the joint model),
DATA_ROOT/ar_reference/<version>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows, LanguageSampler
from diff_voyn.model.ar_reference import CharARLM, ar_loss, ar_preset

AR_VERSION = "v1"


def cosine_with_warmup(warmup: int, total: int, floor: float = 0.1):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


@torch.no_grad()
def heldout_bits(
    model: CharARLM,
    ids: torch.Tensor,
    device: str,
    batch: int,
    lang_idx: int | None = None,
) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch].to(device)
        lang = (
            None
            if lang_idx is None
            else torch.full((len(chunk),), lang_idx, dtype=torch.long, device=device)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            nats = model.nll_nats(chunk, lang)
        total += nats.double().sum().item()
    model.train()
    return total / ids.numel() / math.log(2.0)


def _save(model: CharARLM, cfg, step: int, bits, path: Path, language) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "model": model.state_dict(),
            "step": step,
            "heldout_bits_per_char": bits,
            "language": language,
        },
        path,
    )


def train_multilingual(args, root: Path, logger=None) -> dict:
    """One language-conditioned reference on the backbone's own data mix."""
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    langs = sorted(LANG_TO_INDEX)
    train = CorpusWindows(
        corpus_dir,
        {l: [d["doc_id"] for d in splits["languages"][l]["train"]] for l in langs},
    )
    heldout = CorpusWindows(
        corpus_dir,
        {l: [d["doc_id"] for d in splits["languages"][l]["heldout"]] for l in langs},
    )
    sampler = LanguageSampler(train.chars, args.temperature)
    cfg = ar_preset(args.preset)
    cfg.multilingual = True
    if args.dropout is not None:
        cfg.dropout = args.dropout
    held_ids = {
        l: torch.from_numpy(heldout.tiled_windows(l, cfg.seq_len).astype(np.int64))
        for l in langs
    }

    torch.manual_seed(args.seed)
    model = CharARLM(cfg).to(args.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(args.warmup, args.steps)
    )
    rng = np.random.default_rng([args.seed, 999])
    out_dir = root / "ar_reference" / args.version / "multilingual"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_chars = sum(train.chars.values())
    print(
        f"[multilingual] {model.n_params()/1e6:.1f}M params; train "
        f"{total_chars:,} chars, sampling weights {sampler.weights_dict()}; "
        f"budget {args.steps*args.batch*cfg.seq_len/1e9:.2f}B chars",
        flush=True,
    )
    curve, best = [], (float("inf"), 0, {})
    t0 = time.time()
    running, n_run = 0.0, 0
    for step in range(1, args.steps + 1):
        batch_langs = sampler.sample(rng, size=args.batch)
        ids = torch.from_numpy(
            np.stack(
                [train.sample_window(l, cfg.seq_len, rng) for l in batch_langs]
            ).astype(np.int64)
        ).to(args.device, non_blocking=True)
        lang_idx = torch.tensor(
            [LANG_TO_INDEX[l] for l in batch_langs], device=args.device
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
        ):
            loss = ar_loss(model, ids, lang_idx)
        if not torch.isfinite(loss):
            raise RuntimeError(f"[multilingual] non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += loss.item()
        n_run += 1
        if step % args.log_every == 0:
            bits = running / n_run / math.log(2.0)
            print(
                f"[multilingual] step {step:5d}  train {bits:.3f} bits/char  "
                f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s",
                flush=True,
            )
            if logger:
                logger.report_scalar(
                    "ar_train_bits_per_char", "multilingual", bits, step
                )
            running, n_run = 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            hb = {
                l: heldout_bits(
                    model, held_ids[l], args.device, args.eval_batch, LANG_TO_INDEX[l]
                )
                for l in langs
            }
            mean_bits = float(np.mean(list(hb.values())))
            curve.append((step, mean_bits, hb))
            print(
                f"[multilingual] step {step:5d}  heldout "
                + "  ".join(f"{l} {b:.4f}" for l, b in hb.items())
                + f"  mean {mean_bits:.4f}",
                flush=True,
            )
            if logger:
                for l, b in hb.items():
                    logger.report_scalar("ar_heldout_bits_per_char", f"ml-{l}", b, step)
                logger.report_scalar(
                    "ar_heldout_bits_per_char", "ml-mean", mean_bits, step
                )
            if mean_bits < best[0]:
                best = (mean_bits, step, hb)
                _save(model, cfg, step, hb, out_dir / "ar_best.pt", "multilingual")
    final = curve[-1]
    _save(model, cfg, args.steps, final[2], out_dir / "ar_final.pt", "multilingual")
    return {
        "preset": args.preset,
        "multilingual": True,
        "config": asdict(cfg),
        "n_params": model.n_params(),
        "steps": args.steps,
        "batch_windows": args.batch,
        "sampling_temperature": args.temperature,
        "language_weights": sampler.weights_dict(),
        "train_chars": dict(train.chars),
        "heldout_windows": {l: len(v) for l, v in held_ids.items()},
        "final_heldout_bits_per_char": final[2],
        "best_heldout_bits_per_char": best[2],
        "best_mean_bits": best[0],
        "best_step": best[1],
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
        "best_path": str(out_dir / "ar_best.pt"),
        "final_path": str(out_dir / "ar_final.pt"),
    }


def train_language(lang: str, args, root: Path, logger=None) -> dict:
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    sp = splits["languages"][lang]
    train = CorpusWindows(corpus_dir, {lang: [d["doc_id"] for d in sp["train"]]})
    heldout = CorpusWindows(corpus_dir, {lang: [d["doc_id"] for d in sp["heldout"]]})
    cfg = ar_preset(args.preset)
    if args.dropout is not None:
        cfg.dropout = args.dropout
    held_ids = torch.from_numpy(
        heldout.tiled_windows(lang, cfg.seq_len).astype(np.int64)
    )

    torch.manual_seed(args.seed)
    model = CharARLM(cfg).to(args.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(args.warmup, args.steps)
    )
    rng = np.random.default_rng([args.seed, LANG_TO_INDEX[lang]])
    out_dir = root / "ar_reference" / args.version / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{lang}] {model.n_params()/1e6:.1f}M params; train {train.chars[lang]:,} "
        f"chars ({len(train.docs[lang])} docs); held-out {held_ids.numel():,} chars "
        f"in {len(held_ids)} windows; budget {args.steps*args.batch*cfg.seq_len/1e6:.0f}M "
        f"chars = {args.steps*args.batch*cfg.seq_len/train.chars[lang]:.1f} epochs",
        flush=True,
    )
    curve, best = [], (float("inf"), 0)
    t0 = time.time()
    running, n_run = 0.0, 0
    for step in range(1, args.steps + 1):
        ids = torch.from_numpy(
            np.stack(
                [train.sample_window(lang, cfg.seq_len, rng) for _ in range(args.batch)]
            ).astype(np.int64)
        ).to(args.device, non_blocking=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
        ):
            loss = ar_loss(model, ids)
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{lang}] non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += loss.item()
        n_run += 1
        if step % args.log_every == 0:
            bits = running / n_run / math.log(2.0)
            print(
                f"[{lang}] step {step:5d}  train {bits:.3f} bits/char  "
                f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s",
                flush=True,
            )
            if logger:
                logger.report_scalar("ar_train_bits_per_char", lang, bits, step)
            running, n_run = 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            hb = heldout_bits(model, held_ids, args.device, args.eval_batch)
            curve.append((step, hb))
            print(f"[{lang}] step {step:5d}  heldout {hb:.4f} bits/char", flush=True)
            if logger:
                logger.report_scalar("ar_heldout_bits_per_char", lang, hb, step)
            if hb < best[0]:
                best = (hb, step)
                torch.save(
                    {
                        "config": asdict(cfg),
                        "model": model.state_dict(),
                        "step": step,
                        "heldout_bits_per_char": hb,
                        "language": lang,
                    },
                    out_dir / "ar_best.pt",
                )
    final_bits = curve[-1][1]
    torch.save(
        {
            "config": asdict(cfg),
            "model": model.state_dict(),
            "step": args.steps,
            "heldout_bits_per_char": final_bits,
            "language": lang,
        },
        out_dir / "ar_final.pt",
    )
    return {
        "preset": args.preset,
        "config": asdict(cfg),
        "n_params": model.n_params(),
        "steps": args.steps,
        "batch_windows": args.batch,
        "train_chars": train.chars[lang],
        "epochs": args.steps * args.batch * cfg.seq_len / train.chars[lang],
        "heldout_windows": len(held_ids),
        "heldout_chars": int(held_ids.numel()),
        "final_heldout_bits_per_char": final_bits,
        "best_heldout_bits_per_char": best[0],
        "best_step": best[1],
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
        "best_path": str(out_dir / "ar_best.pt"),
        "final_path": str(out_dir / "ar_final.pt"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", nargs="*", default=list(LANG_TO_INDEX))
    p.add_argument(
        "--multilingual",
        action="store_true",
        help="train ONE language-conditioned model on the backbone's mix (v3)",
    )
    p.add_argument("--temperature", type=float, default=0.7, help="τ (multilingual)")
    p.add_argument("--preset", default="10m")
    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="override preset dropout (small-corpus languages overfit at 0.1)",
    )
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64, help="windows per step")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--version", default=AR_VERSION, help="artifact version dir")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    root = data_root()
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")

    task = logger = None
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task
        from diff_voyn.infra.config import RunConfig

        task = init_task(
            RunConfig(run_name=f"ar-reference-{args.version}", phase="phase3"),
            root,
            tags=["task3.4", args.preset]
            + (["multilingual"] if args.multilingual else []),
        )
        task.connect(vars(args), name="args")
        logger = task.get_logger()

    out_dir = root / "ar_reference" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if args.multilingual:
        summary["multilingual"] = train_multilingual(args, root, logger)
        summary["multilingual"]["created_utc"] = datetime.now(UTC).isoformat()
        summary["multilingual"]["args"] = vars(args)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            "[multilingual] done: best "
            + json.dumps(summary["multilingual"]["best_heldout_bits_per_char"])
            + f" @ {summary['multilingual']['best_step']}"
        )
        args.lang = []
    for lang in args.lang:
        summary[lang] = train_language(lang, args, root, logger)
        summary[lang]["created_utc"] = datetime.now(UTC).isoformat()
        summary[lang]["args"] = vars(args)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            f"[{lang}] done: best {summary[lang]['best_heldout_bits_per_char']:.4f} "
            f"@ {summary[lang]['best_step']}  final "
            f"{summary[lang]['final_heldout_bits_per_char']:.4f} bits/char",
            flush=True,
        )
    print(f"wrote {summary_path}")
    if task:
        task.get_logger().flush(wait=True)
        task.close()


if __name__ == "__main__":
    main()
