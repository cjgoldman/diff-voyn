"""Backbone training runner — tasks 1.3 (pilot) and 1.4 (Phase A).

Design §7.5 defaults: AdamW β₂=0.98, peak LR 3e-4, warmup + cosine (to 10%),
batch ~0.5M chars (micro-batch × grad accumulation), bf16 autocast, dropout
0.1, EMA 0.9999 (the EMA weights are what every eval uses). All languages
jointly from step one (design §7.1), τ=0.7 temperature-balanced sampling.

Per-language held-out NELBO — the canary consulted at every gate — is logged
to ClearML every eval step from the EMA weights, with common random numbers
across language conditions (same masking realizations, design §5a), alongside
the unconditional (NULL-language) NELBO so the task-1.2 acceptance check
(conditional < unconditional) is visible on the dashboard. The realized
language-conditioning dropout rate is logged too (task 1.2).

Usage:
    uv run python scripts/train.py --phase pilot   --model 25m   # task 1.3
    uv run python scripts/train.py --phase phase_a --model 85m   # task 1.4
    ... --resume        # continue from <run_dir>/ckpt_last.pt
    ... --resume --steps 23000 --schedule-total 20000 --ema-reset --ema-decay 0.999
                        # post-G1 EMA tail: hold the LR floor, restart the EMA
                        # from the raw weights with a decay matched to the tail

The canary is logged from the EMA weights *and* from the raw weights: with
decay 0.9999 the EMA lags the model by ~10k steps, so a raw-vs-EMA gap on the
dashboard is what distinguishes genuine improvement from EMA catch-up (the G1
judgement call of 2026-08-20). On ``--resume`` the data stream is re-seeded
from the resume step so an extension never replays the first windows.

Checkpoints and manifests land under DATA_ROOT/runs/<run_name>/.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import (
    LANG_TO_INDEX,
    NULL_LANG_INDEX,
    CorpusWindows,
    DiffVoynIterableDataset,
    LanguageSampler,
)
from diff_voyn.infra.checkpoint import load_checkpoint, save_checkpoint
from diff_voyn.infra.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    model_preset,
)
from diff_voyn.infra.ema import EMA
from diff_voyn.infra.manifest import build_run_manifest, write_run_manifest
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char
from diff_voyn.model.backbone import Backbone, language_dropout_rate
from diff_voyn.model.diffusion import LN2, mdlm_loss

# Fixed seeds shared with scripts/score_checkpoint.py so canary curves and
# offline scores are computed on identical windows and masking realizations.
EVAL_WINDOW_SEED = 12345
EVAL_NELBO_SEED = 0

PHASE_DEFAULTS = {
    # short-budget plumbing test; weights are discarded (task 1.3)
    "pilot": {
        "steps": 2000,
        "accum": 1,
        "warmup": 200,
        "eval_every": 250,
        "ckpt_every": 250,
    },
    # full pretraining until the per-language canary plateaus (task 1.4);
    # accum 16 × micro 32 × 1024 chars ≈ the design's ~0.5M-char batch
    "phase_a": {
        "steps": 20000,
        "accum": 16,
        "warmup": 2000,
        "eval_every": 200,
        "ckpt_every": 200,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["25m", "85m"], default="25m")
    p.add_argument("--phase", choices=list(PHASE_DEFAULTS), default="pilot")
    p.add_argument("--steps", type=int, default=None, help="optimizer steps")
    p.add_argument(
        "--schedule-total",
        type=int,
        default=None,
        help="cosine horizon (default: --steps); with --steps beyond it the LR "
        "holds at the floor — use when extending a finished run",
    )
    p.add_argument("--micro-batch", type=int, default=32, help="sequences per fwd")
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-strata", type=int, default=32)
    p.add_argument("--eval-windows", type=int, default=8, help="per language")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--ema-reset",
        action="store_true",
        help="on --resume, restart the EMA shadow from the raw weights",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="override the EMA decay (default: config 0.9999); with --resume the "
        "new decay applies from the resume step",
    )
    p.add_argument(
        "--no-eval-raw",
        action="store_true",
        help="skip the raw-weight canary (EMA-only, as in the original Phase A)",
    )
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for key, val in PHASE_DEFAULTS[args.phase].items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    if args.schedule_total is None:
        args.schedule_total = args.steps
    if args.ema_reset and not args.resume:
        p.error("--ema-reset only makes sense with --resume")
    if args.run_name is None:
        args.run_name = f"{args.phase}-{args.model}-seed{args.seed}"
    return args


def cosine_with_warmup(warmup: int, total: int, floor: float = 0.1):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


@torch.no_grad()
def canary_eval(
    eval_model: Backbone,
    ema: EMA | None,
    heldout_ids: dict[str, torch.Tensor],
    n_strata: int,
    device: str,
    *,
    raw_model: Backbone | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-language held-out NELBO, CRN across languages.

    Scores the EMA weights (``ema``) — or, with ``raw_model`` given and
    ``ema=None``, the current raw weights copied into ``eval_model`` so
    dropout stays off and the training module is untouched.
    """
    if ema is not None:
        ema.copy_to(eval_model)
    else:
        assert raw_model is not None
        eval_model.load_state_dict(raw_model.state_dict())
    eval_model.eval()
    cond, uncond = {}, {}
    for lang, ids in heldout_ids.items():
        common = {"n_strata": n_strata, "seed": EVAL_NELBO_SEED, "device": device}
        cond[lang] = estimate_nelbo_bits_per_char(
            eval_model, ids, LANG_TO_INDEX[lang], **common
        )
        uncond[lang] = estimate_nelbo_bits_per_char(
            eval_model, ids, NULL_LANG_INDEX, **common
        )
    return cond, uncond


def main() -> None:
    args = parse_args()
    device = args.device
    root = data_root()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model_cfg: ModelConfig = model_preset(args.model)
    cfg = RunConfig(
        run_name=args.run_name,
        phase=args.phase,
        seed=args.seed,
        model=model_cfg,
        data=DataConfig(),
        optim=OptimConfig(
            lr=args.lr,
            warmup_steps=args.warmup,
            batch_chars=args.micro_batch * args.accum * model_cfg.seq_len,
            **({"ema_decay": args.ema_decay} if args.ema_decay is not None else {}),
        ),
    )
    run_dir = root / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---------------------------------------------------------------
    corpus_dir = root / "corpora" / cfg.data.corpus_version
    splits = load_splits(corpus_dir, cfg.data.splits_version)
    train_ids = {
        lang: [d["doc_id"] for d in sp["train"]]
        for lang, sp in splits["languages"].items()
    }
    heldout_doc_ids = {
        lang: [d["doc_id"] for d in sp["heldout"]]
        for lang, sp in splits["languages"].items()
    }
    windows = CorpusWindows(corpus_dir, train_ids)
    weights = LanguageSampler(
        windows.chars, cfg.data.sampling_temperature
    ).weights_dict()

    # Fixed held-out windows for the canary — deterministic, identical across
    # runs and shared with score_checkpoint.py.
    heldout_windows = CorpusWindows(corpus_dir, heldout_doc_ids)
    eval_rng = np.random.default_rng(EVAL_WINDOW_SEED)
    heldout_batch = {
        lang: torch.from_numpy(
            np.stack(
                [
                    heldout_windows.sample_window(lang, model_cfg.seq_len, eval_rng)
                    for _ in range(args.eval_windows)
                ]
            ).astype(np.int64)
        )
        for lang in LANG_TO_INDEX
    }

    # --- model / optimizer ----------------------------------------------------
    model = Backbone(model_cfg).to(device)
    eval_model = Backbone(model_cfg).to(device).eval()
    ema = EMA(model, cfg.optim.ema_decay)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, cosine_with_warmup(args.warmup, args.schedule_total)
    )
    print(f"{args.model} backbone: {model.n_params()/1e6:.1f}M params on {device}")

    start_step = 0
    ckpt_path = run_dir / "ckpt_last.pt"
    if args.resume:
        state = load_checkpoint(
            ckpt_path, model=model, optimizer=opt, scheduler=sched, ema=ema
        )
        start_step = state["step"]
        print(f"resumed from {ckpt_path} at step {start_step}")
        if args.ema_decay is not None:
            ema.decay = args.ema_decay
        if args.ema_reset:
            ema = EMA(model, ema.decay)
        print(
            f"  ema decay {ema.decay}"
            + ("  (shadow reset to raw weights)" if args.ema_reset else "")
            + f"  lr {sched.get_last_lr()[0]:.2e}"
        )
    schedule_info = {
        "steps": args.steps,
        "schedule_total": args.schedule_total,
        "ema_decay": ema.decay,
        "ema_reset_at": start_step if args.ema_reset else None,
        "resumed_from_step": start_step if args.resume else None,
    }

    # Data stream: re-seeded from the resume step so an extension draws fresh
    # windows instead of replaying the stream from step 0.
    data_seed = args.seed + start_step
    dataset = DiffVoynIterableDataset(
        windows,
        seq_len=model_cfg.seq_len,
        temperature=cfg.data.sampling_temperature,
        seed=data_seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=2,
        persistent_workers=True,
        pin_memory=(device == "cuda"),
    )

    manifest = build_run_manifest(cfg, root, language_weights=weights)
    manifest["schedule"] = schedule_info
    manifest["data_seed"] = data_seed
    write_run_manifest(manifest, run_dir)

    task = None
    if not args.no_clearml:
        from diff_voyn.infra.clearml_task import (
            init_task,
            report_language_weights,
            report_per_language_nelbo,
        )

        task = init_task(
            cfg,
            root,
            tags=[args.model, "task1.3" if args.phase == "pilot" else "task1.4"]
            + (["resume"] if args.resume else []),
        )
        task.connect_configuration(schedule_info, name="schedule")
        report_language_weights(task, weights)
        logger = task.get_logger()

    # --- loop -----------------------------------------------------------------
    model.train()
    step = start_step
    running_loss, running_drop, n_running = 0.0, 0.0, 0
    micro = 0
    chars_since = 0
    t_last = time.time()
    opt.zero_grad(set_to_none=True)

    for batch in loader:
        z_t = batch["z_t"].to(device, non_blocking=True)
        ids = batch["ids"].to(device, non_blocking=True)
        masked = batch["mask"].to(device, non_blocking=True)
        t = batch["t"].to(device, non_blocking=True)
        lang_idx = batch["lang_idx"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            logits = model(z_t, lang_idx)
        loss = mdlm_loss(logits, ids, masked, t)
        if not torch.isfinite(loss):
            save_checkpoint(
                run_dir / "ckpt_nonfinite.pt",
                model=model,
                optimizer=opt,
                scheduler=sched,
                ema=ema,
                step=step,
                extra={"config": asdict(cfg), "schedule": schedule_info},
            )
            raise RuntimeError(
                f"non-finite loss at step {step} (see ckpt_nonfinite.pt)"
            )
        (loss / args.accum).backward()

        running_loss += loss.item()
        running_drop += language_dropout_rate(model)
        n_running += 1
        chars_since += ids.numel()
        micro += 1
        if micro < args.accum:
            continue
        micro = 0

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model)
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t_last
            bits = running_loss / n_running / LN2
            drop = running_drop / n_running
            lr = sched.get_last_lr()[0]
            cps = chars_since / dt
            print(
                f"step {step:6d}  train NELBO {bits:6.3f} bits/char  "
                f"lr {lr:.2e}  grad {float(grad_norm):.2f}  "
                f"lang-drop {drop:.3f}  {cps/1e3:.0f}k chars/s",
                flush=True,
            )
            if task:
                logger.report_scalar("train", "nelbo_bits_per_char", bits, step)
                logger.report_scalar("train", "lr", lr, step)
                logger.report_scalar("train", "grad_norm", float(grad_norm), step)
                logger.report_scalar("train", "lang_cond_dropout_rate", drop, step)
                logger.report_scalar("train", "chars_per_sec", cps, step)
            running_loss = running_drop = 0.0
            n_running = 0
            chars_since = 0
            t_last = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            cond, uncond = canary_eval(
                eval_model, ema, heldout_batch, args.eval_strata, device
            )
            model.train()
            msg = "  ".join(
                f"{lang} {cond[lang]:.3f}|{uncond[lang]:.3f}u" for lang in cond
            )
            print(
                f"step {step:6d}  heldout NELBO (EMA, cond|uncond): {msg}", flush=True
            )
            if task:
                report_per_language_nelbo(task, cond, iteration=step)
                for lang, bits in uncond.items():
                    logger.report_scalar(
                        "heldout_nelbo_bits_per_char_unconditional", lang, bits, step
                    )
            if not args.no_eval_raw:
                rcond, runcond = canary_eval(
                    eval_model,
                    None,
                    heldout_batch,
                    args.eval_strata,
                    device,
                    raw_model=model,
                )
                model.train()
                msg = "  ".join(
                    f"{lang} {rcond[lang]:.3f}|{runcond[lang]:.3f}u" for lang in rcond
                )
                print(
                    f"step {step:6d}  heldout NELBO (raw, cond|uncond): {msg}",
                    flush=True,
                )
                if task:
                    for lang in rcond:
                        logger.report_scalar(
                            "heldout_nelbo_bits_per_char_raw", lang, rcond[lang], step
                        )
                        logger.report_scalar(
                            "heldout_nelbo_bits_per_char_raw_unconditional",
                            lang,
                            runcond[lang],
                            step,
                        )
            t_last = time.time()

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=sched,
                ema=ema,
                step=step,
                extra={"config": asdict(cfg), "schedule": schedule_info},
            )

        if step >= args.steps:
            break

    save_checkpoint(
        run_dir / "ckpt_final.pt",
        model=model,
        optimizer=opt,
        scheduler=sched,
        ema=ema,
        step=step,
        extra={"config": asdict(cfg), "schedule": schedule_info},
    )
    print(f"done at step {step}; checkpoints in {run_dir}")
    if task:
        task.get_logger().flush(wait=True)
        print(f"ClearML: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
