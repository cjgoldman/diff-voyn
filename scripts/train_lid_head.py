"""Task 4.2 — attach the language-ID head in Phase B behind a stop-gradient.

The Phase-B backbone (EMA weights, ``--ckpt``) is frozen and never receives
a gradient (features are computed under ``torch.no_grad``; task 4.1
acceptance "backbone grads exactly zero" is enforced by construction and
tested in ``tests/test_lid_head.py``). Only the head
(:class:`diff_voyn.model.lid_head.LIDHead`) trains, on the stream of
:class:`diff_voyn.data.abstain.LIDExampleStream`: clean + noised + NULL-framed
language windows (the Phase-B noise mixture) labelled with their language,
and voynichesque / shuffled windows labelled *abstain* (task 4.3), at window
lengths 128–1024.

Held-out canary every ``--eval-every`` steps on fixed sets (EMA head):
clean / noised / framed accuracy per length and the abstain trigger rate on
voynichesque and shuffled text. Acceptance (4.1 / 4.3): clean long text
>99%, abstain on negative controls >95%. The severity curves of 4.2 are
produced afterwards by ``scripts/lid_eval.py`` on the saved head.

Usage:
    uv run python scripts/train_lid_head.py \\
        --ckpt DATA_ROOT/runs/phase_b-85m-seed0/ckpt_final.pt
Writes DATA_ROOT/runs/lid_head-<model>-seed<seed>/{ckpt_last,lid_head_final}.pt
and summary.json; ClearML tag ``task4.2``.
"""

from __future__ import annotations

import argparse
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
from diff_voyn.data.abstain import (
    LID_KIND_NAMES,
    LIDDataConfig,
    LIDExampleStream,
    build_lid_eval_set,
    load_or_build_voynichesque_pool,
)
from diff_voyn.data.loader import CorpusWindows
from diff_voyn.infra.checkpoint import load_backbone, load_checkpoint, save_checkpoint
from diff_voyn.infra.config import ModelConfig, RunConfig
from diff_voyn.infra.ema import EMA
from diff_voyn.model.lid_head import (
    LID_MASK_LEVELS,
    LIDHead,
    LIDHeadConfig,
    lid_logits,
    lid_loss,
    predict,
)

EVAL_LENGTHS = (128, 256, 1024)


def cosine_with_warmup(warmup: int, total: int, floor: float = 0.1):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_b-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--mask-levels", type=float, nargs="+", default=list(LID_MASK_LEVELS)
    )
    p.add_argument("--p-abstain", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-n", type=int, default=32, help="windows per language")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return args


@torch.no_grad()
def evaluate(backbone, head, ema, eval_sets, device, batch=32) -> dict[str, float]:
    """Accuracy per fixed held-out set with the EMA head."""
    eval_head = LIDHead(head.cfg).to(device)
    ema.copy_to(eval_head)
    out = {}
    for name, (ids, labels) in eval_sets.items():
        probs = predict(
            backbone,
            eval_head,
            ids,
            batch=batch,
            seed=0,
            device=device,
            calibrated=False,
        )
        out[name] = float((probs.argmax(-1) == labels).float().mean())
    return out


def main() -> None:
    args = parse_args()
    device = args.device
    root = data_root()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    backbone, bmeta = load_backbone(args.ckpt, device, ema=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    model_cfg = ModelConfig(**bmeta["model"])
    size = "85m" if model_cfg.d_model == 768 else "25m"
    run_name = args.run_name or f"lid_head-{size}-seed{args.seed}"
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"backbone {args.ckpt} (step {bmeta['step']}, {bmeta['weights']} weights, "
        f"{backbone.n_params()/1e6:.1f}M params) frozen on {device}"
    )

    # --- data -----------------------------------------------------------------
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    train_ids = {
        l: [d["doc_id"] for d in sp["train"]] for l, sp in splits["languages"].items()
    }
    heldout_ids = {
        l: [d["doc_id"] for d in sp["heldout"]] for l, sp in splits["languages"].items()
    }
    train_windows = CorpusWindows(corpus_dir, train_ids)
    heldout_windows = CorpusWindows(corpus_dir, heldout_ids)
    t0 = time.time()
    pool_train = load_or_build_voynichesque_pool(root, train_windows, "train")
    pool_heldout = load_or_build_voynichesque_pool(
        root, heldout_windows, "heldout", n_encryptions=120, seed=1
    )
    print(
        f"voynichesque pools: train {len(pool_train)} encryptions "
        f"({sum(map(len, pool_train))/1e6:.2f}M chars), heldout {len(pool_heldout)} "
        f"({time.time()-t0:.0f}s)"
    )
    data_cfg = LIDDataConfig(p_abstain=args.p_abstain, batch=args.batch)
    stream = LIDExampleStream(train_windows, pool_train, data_cfg, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=2, persistent_workers=True
    )
    eval_sets = build_lid_eval_set(
        heldout_windows, pool_heldout, n_per_language=args.eval_n, lengths=EVAL_LENGTHS
    )

    # --- head -----------------------------------------------------------------
    head_cfg = LIDHeadConfig(
        d_model=model_cfg.d_model,
        hidden=args.hidden,
        dropout=args.dropout,
        mask_levels=tuple(args.mask_levels),
    )
    head = LIDHead(head_cfg).to(device)
    ema = EMA(head, args.ema_decay)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(args.warmup, args.steps)
    )
    print(f"LID head: {head.n_params()/1e3:.0f}k params, levels {head_cfg.mask_levels}")
    extra = {
        "lid_head_config": head_cfg.to_dict(),
        "backbone": bmeta,
        "config": {
            "phase": "phase_b_lid",
            "run_name": run_name,
            "model": bmeta["model"],
        },
        "data": data_cfg.to_dict(),
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
    }
    start_step = 0
    ckpt_path = run_dir / "ckpt_last.pt"
    if args.resume:
        st = load_checkpoint(
            ckpt_path, model=head, optimizer=opt, scheduler=sched, ema=ema
        )
        start_step = st["step"]
        print(f"resumed at step {start_step}")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"created_utc": datetime.now(UTC).isoformat(), **extra}, indent=2)
    )

    task = None
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import init_task

        task = init_task(
            RunConfig(
                run_name=run_name, phase="phase4", seed=args.seed, model=model_cfg
            ),
            root,
            tags=[size, "task4.2", "task4.3", "stop-gradient"],
        )
        task.connect_configuration(extra, name="lid_head")
        logger = task.get_logger()

    # --- loop -----------------------------------------------------------------
    step = start_step
    n_kinds = max(LID_KIND_NAMES) + 1
    kind_correct = torch.zeros(n_kinds, dtype=torch.float64)
    kind_count = torch.zeros(n_kinds, dtype=torch.float64)
    run_loss, run_n = 0.0, 0
    t_last = time.time()
    head.train()
    for batch in loader:
        if step >= args.steps:
            break
        ids = batch["ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        g = torch.Generator().manual_seed(args.seed * 100_003 + step)
        logits = lid_logits(backbone, head, ids, g=g, stop_gradient=True, autocast=True)
        loss = lid_loss(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(head)
        step += 1

        correct = (logits.argmax(-1) == labels).double().cpu()
        kind_correct.index_add_(0, batch["kind"], correct)
        kind_count.index_add_(0, batch["kind"], torch.ones_like(correct))
        run_loss += float(loss.detach())
        run_n += 1

        if step % args.log_every == 0:
            acc_by_kind = {
                LID_KIND_NAMES[k]: float(kind_correct[k] / kind_count[k])
                for k in range(n_kinds)
                if kind_count[k] > 0
            }
            acc = float(kind_correct.sum() / kind_count.sum())
            dt = time.time() - t_last
            print(
                f"step {step:5d}  loss {run_loss/run_n:.4f}  acc {acc:.3f}  "
                + "  ".join(f"{k} {v:.2f}" for k, v in acc_by_kind.items())
                + f"  lr {sched.get_last_lr()[0]:.1e}  {dt/args.log_every:.2f}s/step",
                flush=True,
            )
            if task:
                logger.report_scalar("lid_train", "loss", run_loss / run_n, step)
                logger.report_scalar("lid_train", "acc", acc, step)
                logger.report_scalar("lid_train", "lr", sched.get_last_lr()[0], step)
                for k, v in acc_by_kind.items():
                    logger.report_scalar("lid_train_acc_by_kind", k, v, step)
            kind_correct.zero_()
            kind_count.zero_()
            run_loss, run_n = 0.0, 0
            t_last = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(backbone, head, ema, eval_sets, device)
            head.train()
            print(
                f"step {step:5d}  heldout LID acc (EMA): "
                + "  ".join(f"{k} {v:.3f}" for k, v in res.items()),
                flush=True,
            )
            if task:
                for k, v in res.items():
                    logger.report_scalar("lid_heldout_acc", k, v, step)
            t_last = time.time()

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(
                ckpt_path,
                model=head,
                optimizer=opt,
                scheduler=sched,
                ema=ema,
                step=step,
                extra=extra,
            )

    final = evaluate(backbone, head, ema, eval_sets, device)
    save_checkpoint(
        run_dir / "lid_head_final.pt",
        model=head,
        optimizer=opt,
        scheduler=sched,
        ema=ema,
        step=step,
        extra=extra,
    )
    clean_long = final["clean_L1024"]
    abstain = {
        k: v
        for k, v in final.items()
        if k.startswith(("voynichesque", "shuffled", "uniform"))
    }
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "run_name": run_name,
        "step": step,
        "backbone": bmeta,
        "lid_head_config": head_cfg.to_dict(),
        "data": data_cfg.to_dict(),
        "heldout_acc": final,
        "acceptance": {
            "4.1 clean long text > 99%": {
                "value": clean_long,
                "pass": clean_long > 0.99,
            },
            "4.3 abstain on negative controls > 95%": {
                "value": abstain,
                "pass": all(v > 0.95 for v in abstain.values()),
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["acceptance"], indent=1))
    print(f"done at step {step}; head in {run_dir / 'lid_head_final.pt'}")
    if task:
        task.connect_configuration(summary, name="summary")
        for k, v in final.items():
            logger.report_scalar("lid_heldout_acc_final", k, v, step)
        logger.flush(wait=True)
        print(f"ClearML: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
