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
    uv run python scripts/train.py --phase phase_b --model 85m \
        --init-from DATA_ROOT/runs/phase_a-85m-seed0/ckpt_final.pt   # task 2.4/2.5
    ... --resume        # continue from <run_dir>/ckpt_last.pt
    ... --resume --steps 23000 --schedule-total 20000 --ema-reset --ema-decay 0.999
                        # post-G1 EMA tail: hold the LR floor, restart the EMA
                        # from the raw weights with a decay matched to the tail

The canary is logged from the EMA weights *and* from the raw weights: with
decay 0.9999 the EMA lags the model by ~10k steps, so a raw-vs-EMA gap on the
dashboard is what distinguishes genuine improvement from EMA catch-up (the G1
judgement call of 2026-08-20). On ``--resume`` the data stream is re-seeded
from the resume step so an extension never replays the first windows.

Phase B (tasks 2.4 + 2.5, design §7.3/§8) fine-tunes the Phase-A weights
(EMA shadow of ``--init-from``, fresh optimizer, lower peak LR, EMA decay
matched to the run length per the G1 lesson) on the noise mixture of
``diff_voyn.data.noise.NoiseConfig``: 50% clean (the calibration anchor,
never reduced), 40% simulated partial decipherments, 20% on the 2N-slot NULL
frame. The canary gains fixed noised and NULL-framed held-out variants so the
G2 criteria (clean anchor held, noised NELBO degrading smoothly, NULL slots
in-distribution) are on the dashboard while the run is live. Realized
example-kind fractions and per-kind train NELBO are logged every
``--log-every`` steps.

Phase C (tasks 4.4 + 4.5, design §7.2) releases the language-ID head's
stop-gradient: ``L = L_NELBO + λ·L_LID`` on the Phase-B weights
(``--init-from``) with the Phase-B-trained head (``--lid-head``), λ ramped
0 → ``--lid-lambda-max`` (0.05) over ``--lid-ramp-steps``. Two guards, both
logged and recorded in the run manifest / ``lambda_schedule.json``:
(a) if the LID gradient on the backbone exceeds ``--lid-grad-ratio-max``
(10%) of the diffusion gradient, the λ cap is halved; (b) the per-language
held-out canary (EMA *and* raw weights) is compared with its value at the
start of Phase C and the cap is halved when any language degrades by more
than ``--canary-degrade`` (1%) relative. The LID loss uses its own batch
stream (clean / noised / framed language windows + abstain windows of
``diff_voyn.data.abstain``); abstain text never enters the diffusion loss.
The noise mixture and the clean fraction are retained (design §7.3).

Checkpoints and manifests land under DATA_ROOT/runs/<run_name>/.
"""

from __future__ import annotations

import argparse
import json
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
from diff_voyn.data.abstain import (
    LIDDataConfig,
    LIDExampleStream,
    build_lid_eval_set,
    load_or_build_voynichesque_pool,
)
from diff_voyn.data.loader import (
    LANG_TO_INDEX,
    NULL_LANG_INDEX,
    CorpusWindows,
    DiffVoynIterableDataset,
    LanguageSampler,
)
from diff_voyn.data.noise import (
    EVAL_NOISE_SEED,
    KIND_NAMES,
    NoiseConfig,
    NoiseMixture,
    framed_variant,
    noised_variant,
)
from diff_voyn.infra.checkpoint import (
    load_backbone,
    load_checkpoint,
    load_lid_head,
    save_checkpoint,
)
from diff_voyn.infra.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    model_preset,
)
from diff_voyn.infra.ema import EMA
from diff_voyn.infra.manifest import build_run_manifest, write_run_manifest
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char, per_position_nelbo_bits
from diff_voyn.model.backbone import Backbone, language_dropout_rate
from diff_voyn.model.diffusion import LN2, mdlm_nelbo_terms
from diff_voyn.model.lid_head import (
    LIDHead,
    backbone_grad_norm,
    lambda_schedule,
    lid_loss,
    mask_at_level,
    pooled_features,
    predict,
)
from diff_voyn.vocab import NULL_ID

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
        "lr": 3e-4,
        "ema_decay": None,  # config default 0.9999
    },
    # full pretraining until the per-language canary plateaus (task 1.4);
    # accum 16 × micro 32 × 1024 chars ≈ the design's ~0.5M-char batch
    "phase_a": {
        "steps": 20000,
        "accum": 16,
        "warmup": 2000,
        "eval_every": 200,
        "ckpt_every": 200,
        "lr": 3e-4,
        "ema_decay": None,
    },
    # noise-curriculum fine-tune from the Phase-A weights (tasks 2.4/2.5):
    # same batch, peak LR 1e-4 (≈3× the Phase-A floor) with a short warmup
    # and cosine to 1e-5; EMA 0.999 so the time constant (1000 steps) is
    # ≤1/5 of the run (the G1 EMA-lag lesson, docs/phase1_status.md)
    "phase_b": {
        "steps": 6000,
        "accum": 16,
        "warmup": 300,
        "eval_every": 200,
        "ckpt_every": 200,
        "lr": 1e-4,
        "ema_decay": 0.999,
    },
    # joint LID fine-tune from the Phase-B weights (tasks 4.4/4.5): short,
    # low peak LR (3e-5 ≈ 3× the Phase-B floor, cosine to 3e-6) so the
    # adjustments stay small and monitorable (design §7.2); EMA 0.9975 so
    # the time constant (400 steps) is ≤1/5 of the run (the G1 lesson)
    "phase_c": {
        "steps": 2000,
        "accum": 16,
        "warmup": 100,
        "eval_every": 100,
        "ckpt_every": 100,
        "lr": 3e-5,
        "ema_decay": 0.9975,
    },
}
PHASE_TAGS = {
    "pilot": "task1.3",
    "phase_a": "task1.4",
    "phase_b": "task2.4",
    "phase_c": "task4.4",
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
    p.add_argument("--lr", type=float, default=None, help="peak LR (phase default)")
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
        help="override the EMA decay (default: phase default / config 0.9999); "
        "with --resume the new decay applies from the resume step",
    )
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="initialize the model from this checkpoint's EMA weights (fresh "
        "optimizer/schedule) — Phase B starts from the Phase-A final",
    )
    p.add_argument(
        "--init-raw",
        action="store_true",
        help="with --init-from, take the raw weights instead of the EMA shadow",
    )
    p.add_argument(
        "--no-noise",
        action="store_true",
        help="phase_b ablation: same schedule, clean data only",
    )
    p.add_argument(
        "--noise-mix",
        type=float,
        nargs=4,
        metavar=("CLEAN", "NOISED", "FRAMED", "FRAMED_NOISED"),
        default=None,
        help="example-kind probabilities (default NoiseConfig: 0.5 0.3 0.1 0.1)",
    )
    p.add_argument(
        "--noise-families",
        nargs="+",
        choices=["substitution", "segmentation", "transcription"],
        default=None,
        help="phase_b ablation (control experiment 6a): restrict the noise "
        "families to this subset (default: all three). E.g. "
        "'--noise-families segmentation transcription' re-runs the curriculum "
        "WITHOUT the wrong-key family",
    )
    p.add_argument(
        "--no-eval-raw",
        action="store_true",
        help="skip the raw-weight canary (EMA-only, as in the original Phase A)",
    )
    # Phase C (tasks 4.4 / 4.5)
    p.add_argument(
        "--lid-head",
        type=Path,
        default=None,
        help="phase_c: the Phase-B-trained head checkpoint (scripts/train_lid_head.py)",
    )
    p.add_argument("--lid-lambda-max", type=float, default=0.05, help="λ cap")
    p.add_argument(
        "--lid-ramp-steps",
        type=int,
        default=None,
        help="linear λ ramp length (default: half of --steps)",
    )
    p.add_argument("--lid-batch", type=int, default=8, help="LID windows per step")
    p.add_argument("--lid-head-lr", type=float, default=1e-4)
    p.add_argument(
        "--lid-grad-ratio-max",
        type=float,
        default=0.10,
        help="halve the λ cap when ‖λ∇L_LID‖/‖∇L_NELBO‖ on the backbone exceeds this",
    )
    p.add_argument(
        "--canary-degrade",
        type=float,
        default=0.01,
        help="halve the λ cap when any language's held-out NELBO degrades by "
        "more than this relative to the start of Phase C",
    )
    p.add_argument("--lid-p-abstain", type=float, default=0.25)
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for key, val in PHASE_DEFAULTS[args.phase].items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    if args.schedule_total is None:
        args.schedule_total = args.steps
    if args.lid_ramp_steps is None:
        args.lid_ramp_steps = args.steps // 2
    if args.ema_reset and not args.resume:
        p.error("--ema-reset only makes sense with --resume")
    if args.phase == "phase_b" and not (args.init_from or args.resume):
        p.error("--phase phase_b needs --init-from <phase_a checkpoint> (or --resume)")
    if (
        args.phase == "phase_c"
        and not args.resume
        and not (args.init_from and args.lid_head)
    ):
        p.error("--phase phase_c needs --init-from <phase_b checkpoint> and --lid-head")
    if args.init_from and args.resume:
        p.error("--init-from and --resume are mutually exclusive")
    if args.run_name is None:
        args.run_name = f"{args.phase}-{args.model}-seed{args.seed}"
    return args


def ckpt_path_for(run_dir: Path) -> Path:
    return run_dir / "ckpt_last.pt"


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


def build_noise_canary(
    heldout_windows: CorpusWindows, seq_len: int, n_windows: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Fixed held-out variants for the Phase-B canary (tasks 2.4/2.5): per
    language, ``clean_ref`` (the first seq_len chars of longer source
    windows), ``noised`` (the moderate fixed-severity partial decipherment of
    ``noise.EVAL_NOISED_SEVERITIES`` applied to those sources) and ``framed``
    (the same sources on the 2N-slot NULL frame). Deterministic: window seed
    EVAL_WINDOW_SEED+1, noise seed EVAL_NOISE_SEED."""
    src_len = int(seq_len * NoiseConfig().source_margin)
    rng = np.random.default_rng(EVAL_WINDOW_SEED + 1)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for lang in LANG_TO_INDEX:
        sources = [
            heldout_windows.sample_window(lang, src_len, rng) for _ in range(n_windows)
        ]
        nrng = np.random.default_rng([EVAL_NOISE_SEED, LANG_TO_INDEX[lang]])
        noised = np.stack([noised_variant(s, seq_len, nrng) for s in sources])
        frng = np.random.default_rng([EVAL_NOISE_SEED, 10 + LANG_TO_INDEX[lang]])
        framed = np.stack([framed_variant(s, seq_len, frng) for s in sources])
        clean = np.stack([s[:seq_len] for s in sources])
        out[lang] = {
            k: torch.from_numpy(v.astype(np.int64))
            for k, v in (("clean_ref", clean), ("noised", noised), ("framed", framed))
        }
    return out


@torch.no_grad()
def noise_canary_eval(
    eval_model: Backbone,
    ema: EMA,
    variants: dict[str, dict[str, torch.Tensor]],
    n_strata: int,
    device: str,
) -> dict[str, dict[str, float]]:
    """EMA-weight NELBO on the fixed noised / framed held-out variants, own
    language, CRN (same masking seed as the clean canary). For the framed
    variant the per-position bound is split into NULL-slot and letter-slot
    means (task 2.5 acceptance: NULL slots in-distribution)."""
    ema.copy_to(eval_model)
    eval_model.eval()
    out: dict[str, dict[str, float]] = {}
    for lang, v in variants.items():
        li = LANG_TO_INDEX[lang]
        common = {"n_strata": n_strata, "seed": EVAL_NELBO_SEED, "device": device}
        r = {
            "clean_ref": estimate_nelbo_bits_per_char(
                eval_model, v["clean_ref"], li, **common
            ),
            "noised": estimate_nelbo_bits_per_char(
                eval_model, v["noised"], li, **common
            ),
        }
        pos = per_position_nelbo_bits(eval_model, v["framed"], li, **common)
        is_null = v["framed"] == NULL_ID
        r["framed_per_slot"] = float(pos.mean())
        r["null_slot_bits"] = float(pos[is_null].mean())
        r["letter_slot_bits"] = float(pos[~is_null].mean())
        out[lang] = r
    return out


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
    noise_cfg: NoiseConfig | None = None
    if args.phase in ("phase_b", "phase_c") and not args.no_noise:
        noise_cfg = NoiseConfig(*args.noise_mix) if args.noise_mix else NoiseConfig()
        if args.noise_families is not None:
            fams = set(args.noise_families)
            noise_cfg = NoiseConfig(
                **{
                    **noise_cfg.to_dict(),
                    "p_substitution": noise_cfg.p_substitution
                    if "substitution" in fams
                    else 0.0,
                    "p_segmentation": noise_cfg.p_segmentation
                    if "segmentation" in fams
                    else 0.0,
                    "p_transcription": noise_cfg.p_transcription
                    if "transcription" in fams
                    else 0.0,
                }
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
    phase_c = args.phase == "phase_c"
    lid_head: LIDHead | None = None
    lid_ema: EMA | None = None
    lid_meta = None
    param_groups = [{"params": list(model.parameters()), "lr": cfg.optim.lr}]
    if phase_c:
        if args.lid_head:
            lid_head, lid_meta = load_lid_head(args.lid_head, device, ema=True)
        else:  # resume: the architecture comes from the checkpoint
            st = torch.load(
                ckpt_path_for(run_dir), map_location="cpu", weights_only=False
            )
            from diff_voyn.model.lid_head import LIDHeadConfig

            lid_head = LIDHead(
                LIDHeadConfig.from_dict(st["extra"]["lid_head_config"])
            ).to(device)
            lid_meta = st["extra"].get("lid_head_source")
        lid_head.train()
        lid_ema = EMA(lid_head, cfg.optim.ema_decay)
        param_groups.append(
            {"params": list(lid_head.parameters()), "lr": args.lid_head_lr}
        )
    opt = torch.optim.AdamW(
        param_groups,
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
    init_meta = None
    if args.init_from:
        init_model, init_meta = load_backbone(
            args.init_from, device, ema=not args.init_raw
        )
        model.load_state_dict(init_model.state_dict())
        del init_model
        ema = EMA(model, cfg.optim.ema_decay)  # shadow starts at the loaded weights
        print(
            f"initialized from {args.init_from} (step {init_meta['step']}, "
            f"{init_meta['weights']} weights); fresh optimizer, ema decay {ema.decay}"
        )
    lambda_max = args.lid_lambda_max
    lambda_events: list[dict] = []
    lambda_trace: list[dict] = []
    canary_ref: dict[str, float] | None = None
    if args.resume:
        state = load_checkpoint(
            ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            ema=ema,
            lid_head=lid_head,
            lid_ema=lid_ema,
        )
        start_step = state["step"]
        pc = state["extra"].get("phase_c") or {}
        lambda_max = pc.get("lambda_max_current", lambda_max)
        lambda_events = pc.get("lambda_events", [])
        lambda_trace = pc.get("lambda_trace", [])
        canary_ref = pc.get("canary_reference")
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
        "init_from": init_meta,
        "peak_lr": cfg.optim.lr,
        "warmup": args.warmup,
    }
    if phase_c:
        schedule_info["lid"] = {
            "head_source": lid_meta,
            "lambda_max_initial": args.lid_lambda_max,
            "ramp_steps": args.lid_ramp_steps,
            "grad_ratio_max": args.lid_grad_ratio_max,
            "canary_degrade": args.canary_degrade,
            "lid_batch": args.lid_batch,
            "lid_head_lr": args.lid_head_lr,
            "p_abstain": args.lid_p_abstain,
        }

    # Data stream: re-seeded from the resume step so an extension draws fresh
    # windows instead of replaying the stream from step 0.
    data_seed = args.seed + start_step
    dataset = DiffVoynIterableDataset(
        windows,
        seq_len=model_cfg.seq_len,
        temperature=cfg.data.sampling_temperature,
        seed=data_seed,
        noise=NoiseMixture(noise_cfg) if noise_cfg else None,
    )
    noise_canary = None
    if noise_cfg is not None:
        noise_canary = build_noise_canary(
            heldout_windows, model_cfg.seq_len, args.eval_windows
        )
    lid_iter = None
    lid_eval_sets = None
    if phase_c:
        pool_train = load_or_build_voynichesque_pool(root, windows, "train")
        pool_heldout = load_or_build_voynichesque_pool(
            root, heldout_windows, "heldout", n_encryptions=120, seed=1
        )
        lid_cfg = LIDDataConfig(p_abstain=args.lid_p_abstain, batch=args.lid_batch)
        lid_stream = LIDExampleStream(windows, pool_train, lid_cfg, seed=data_seed + 7)
        lid_iter = iter(
            torch.utils.data.DataLoader(
                lid_stream, batch_size=None, num_workers=1, persistent_workers=True
            )
        )
        lid_eval_sets = build_lid_eval_set(
            heldout_windows, pool_heldout, n_per_language=16, lengths=(256, 1024)
        )
        schedule_info["lid"]["data"] = lid_cfg.to_dict()
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
    manifest["noise"] = noise_cfg.to_dict() if noise_cfg else None
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
            tags=[args.model, PHASE_TAGS[args.phase]]
            + (["resume"] if args.resume else [])
            + (["no-noise"] if args.no_noise else [])
            + (
                ["families:" + "+".join(sorted(args.noise_families))]
                if args.noise_families
                else []
            ),
        )
        task.connect_configuration(schedule_info, name="schedule")
        if phase_c:
            task.connect_configuration(schedule_info["lid"], name="lid")
        if noise_cfg is not None:
            task.connect_configuration(noise_cfg.to_dict(), name="noise")
        report_language_weights(task, weights)
        logger = task.get_logger()

    def phase_c_extra() -> dict:
        return {
            "lid_head_config": lid_head.cfg.to_dict(),
            "lid_head_source": lid_meta,
            "phase_c": {
                "lambda_max_current": lambda_max,
                "lambda_events": lambda_events,
                "lambda_trace": lambda_trace,
                "canary_reference": canary_ref,
            },
        }

    def ckpt_extra() -> dict:
        extra = {"config": asdict(cfg), "schedule": schedule_info}
        if phase_c:
            extra.update(phase_c_extra())
            (run_dir / "lambda_schedule.json").write_text(
                json.dumps(extra["phase_c"], indent=1)
            )
        return extra

    if phase_c and canary_ref is None:
        # task 4.5: the abort criterion is relative to the start of Phase C —
        # same windows, same masking seed, so the comparison is exact.
        canary_ref, _ = canary_eval(
            eval_model, ema, heldout_batch, args.eval_strata, device
        )
        print(
            "phase C canary reference (EMA = Phase-B weights): "
            + "  ".join(f"{l} {v:.4f}" for l, v in canary_ref.items()),
            flush=True,
        )
    if phase_c:
        print(
            f"LID head: {lid_head.n_params()/1e3:.0f}k params from {lid_meta and lid_meta.get('path')}; "
            f"λ ramp 0→{lambda_max} over {args.lid_ramp_steps} steps, head lr {args.lid_head_lr}"
        )

    # --- loop -----------------------------------------------------------------
    model.train()
    step = start_step

    def fresh_lid_running() -> dict:
        return {
            "loss": 0.0,
            "correct": 0.0,
            "n": 0.0,
            "gnorm": 0.0,
            "total_gnorm": 0.0,
            "steps": 0,
        }

    lid_running = fresh_lid_running()
    lid_grad_snapshot = None
    lam = 0.0
    running_loss, running_drop, n_running = 0.0, 0.0, 0
    kind_loss = torch.zeros(4, dtype=torch.float64)
    kind_count = torch.zeros(4, dtype=torch.float64)
    running_sev = 0.0
    micro = 0
    chars_since = 0
    t_last = time.time()
    opt.zero_grad(set_to_none=True)

    for batch in loader:
        if phase_c and micro == 0:
            # Joint objective (task 4.4): the LID loss's gradient reaches the
            # backbone scaled by λ (features = λ·f + (1−λ)·stop_grad(f)), the
            # head always trains on the unscaled loss. Done first in the
            # accumulation window so its backbone gradient can be measured
            # alone (task 4.4: LID grad norm < 10% of the diffusion grad).
            lam = lambda_schedule(step, args.lid_ramp_steps, lambda_max)
            lb = next(lid_iter)
            lid_ids = lb["ids"].to(device, non_blocking=True)
            lid_labels = lb["label"].to(device, non_blocking=True)
            lid_seed = args.seed * 100_003 + step
            # Pass 1 (no autograd through the backbone): pooled features at
            # every masking level → head forward/backward. The head trains on
            # the unscaled loss; the gradient w.r.t. the pooled feature is
            # kept for pass 2.
            feats0 = pooled_features(
                model,
                lid_ids,
                lid_head.cfg.mask_levels,
                g=torch.Generator().manual_seed(lid_seed),
                stop_gradient=True,
                autocast=(device == "cuda"),
            )
            feats_leaf = feats0.detach().requires_grad_(True)
            lid_logits_b = lid_head(feats_leaf)
            l_lid = lid_loss(lid_logits_b, lid_labels)
            l_lid.backward()
            # Pass 2: the backbone receives λ·∂L_LID/∂θ one masking level at
            # a time (same masks as pass 1 via the re-seeded generator), so
            # only one level's graph is alive — the four-level graph of a
            # single backward OOMs the 85M on a 24 GB card. At λ = 0 the
            # backbone gradient is exactly zero (pass skipped).
            if lam > 0:
                g_feat = feats_leaf.grad.detach() * (
                    lam / len(lid_head.cfg.mask_levels)
                )
                g_mask = torch.Generator().manual_seed(lid_seed)
                lid_lang = torch.full(
                    (lid_ids.shape[0],),
                    NULL_LANG_INDEX,
                    dtype=torch.long,
                    device=device,
                )
                for level in lid_head.cfg.mask_levels:
                    z_lid = mask_at_level(lid_ids, level, g_mask)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=(device == "cuda"),
                    ):
                        h_lid = model.hidden(z_lid, lid_lang)
                    h_lid.float().mean(dim=1).backward(g_feat)
                    del h_lid, z_lid
            lid_gnorm = backbone_grad_norm(model)
            lid_running["loss"] += float(l_lid.detach())
            lid_running["correct"] += float(
                (lid_logits_b.argmax(-1) == lid_labels).float().sum()
            )
            lid_running["n"] += float(len(lid_labels))
            lid_running["gnorm"] += lid_gnorm
            lid_running["steps"] += 1
            if (step + 1) % args.log_every == 0:
                lid_grad_snapshot = [
                    (p, p.grad.detach().clone())
                    for p in model.parameters()
                    if p.grad is not None
                ]
        z_t = batch["z_t"].to(device, non_blocking=True)
        ids = batch["ids"].to(device, non_blocking=True)
        masked = batch["mask"].to(device, non_blocking=True)
        t = batch["t"].to(device, non_blocking=True)
        lang_idx = batch["lang_idx"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            logits = model(z_t, lang_idx)
        terms = mdlm_nelbo_terms(logits, ids, masked, t)
        loss = terms.mean()
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
        kinds = batch["kind"]
        kind_loss.index_add_(0, kinds, terms.detach().double().cpu())
        kind_count.index_add_(0, kinds, torch.ones(len(kinds), dtype=torch.float64))
        running_sev += float(batch["sub_severity"].sum())
        chars_since += ids.numel()
        micro += 1
        if micro < args.accum:
            continue
        micro = 0

        grad_ratio = None
        if phase_c and lid_grad_snapshot is not None:
            # exact single-step ratio (diagnostic): LID vs diffusion gradient
            # on the backbone, the latter obtained by difference
            sq_lid, sq_diff = 0.0, 0.0
            for p, g_l in lid_grad_snapshot:
                g_d = p.grad.detach().float() - g_l.float()
                sq_lid += float(g_l.float().pow(2).sum())
                sq_diff += float(g_d.pow(2).sum())
            grad_ratio = (sq_lid**0.5) / max(sq_diff**0.5, 1e-12)
            lid_grad_snapshot = None
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if phase_c:
            lid_running["total_gnorm"] += float(grad_norm)
        if phase_c:
            torch.nn.utils.clip_grad_norm_(lid_head.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model)
        if phase_c:
            lid_ema.update(lid_head)
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
            if phase_c:
                n_l = max(1, lid_running["steps"])
                lid_loss_m = lid_running["loss"] / n_l
                lid_acc_m = lid_running["correct"] / max(1.0, lid_running["n"])
                lid_g_m = lid_running["gnorm"] / n_l
                total_g_m = lid_running["total_gnorm"] / n_l
                # The 10% rule (task 4.4) is judged on window means, not on a
                # single-step snapshot: a 12-window LID batch has a noisy
                # gradient whose norm spikes on hard batches (one spike at
                # step 150 of the first 25M run read 0.74 against a typical
                # 0.02–0.07). Window mean ‖λ∇L_LID‖ over the mean total
                # gradient norm (≥ the diffusion part) — conservative.
                ratio_window = lid_g_m / max(total_g_m, 1e-12)
                rec = {
                    "step": step,
                    "lambda": lam,
                    "lambda_max": lambda_max,
                    "grad_ratio": ratio_window,
                    "grad_ratio_snapshot": grad_ratio,
                    "lid_grad_norm": lid_g_m,
                    "total_grad_norm": total_g_m,
                    "lid_loss": lid_loss_m,
                    "lid_acc": lid_acc_m,
                }
                lambda_trace.append(rec)
                print(
                    f"            LID λ {lam:.4f} (cap {lambda_max:.4f})  loss {lid_loss_m:.4f}  "
                    f"acc {lid_acc_m:.3f}  ‖λ∇_bb L_LID‖ {lid_g_m:.4f}  ratio {ratio_window:.4f} "
                    f"(snapshot {grad_ratio if grad_ratio is None else round(grad_ratio, 4)})",
                    flush=True,
                )
                if lam > 0 and ratio_window > args.lid_grad_ratio_max:
                    lambda_max *= 0.5
                    lambda_events.append(
                        {
                            "step": step,
                            "reason": "grad_ratio",
                            "grad_ratio": ratio_window,
                            "lambda_max": lambda_max,
                        }
                    )
                    print(
                        f"step {step:6d}  LID/diffusion grad ratio {ratio_window:.3f} > "
                        f"{args.lid_grad_ratio_max}: λ cap halved to {lambda_max:.4f}",
                        flush=True,
                    )
                if task:
                    logger.report_scalar("lid", "lambda", lam, step)
                    logger.report_scalar("lid", "lambda_max", lambda_max, step)
                    logger.report_scalar("lid", "train_loss", lid_loss_m, step)
                    logger.report_scalar("lid", "train_acc", lid_acc_m, step)
                    logger.report_scalar("lid", "backbone_grad_norm", lid_g_m, step)
                    logger.report_scalar("lid", "grad_ratio", ratio_window, step)
                    if grad_ratio is not None:
                        logger.report_scalar(
                            "lid", "grad_ratio_snapshot", grad_ratio, step
                        )
                lid_running = fresh_lid_running()
            if noise_cfg is not None:
                n_ex = float(kind_count.sum())
                fracs = (kind_count / n_ex).tolist()
                per_kind = (kind_loss / kind_count.clamp_min(1) / LN2).tolist()
                n_noised = float(kind_count[1] + kind_count[3])
                mean_sev = running_sev / max(1.0, n_noised)
                print(
                    "            kinds "
                    + "  ".join(
                        f"{KIND_NAMES[k]} {fracs[k]:.2f}:{per_kind[k]:.3f}"
                        for k in range(4)
                    )
                    + f"  mean sub-severity {mean_sev:.3f}",
                    flush=True,
                )
                if task:
                    for k in range(4):
                        logger.report_scalar(
                            "train_kind_fraction", KIND_NAMES[k], fracs[k], step
                        )
                        if kind_count[k] > 0:
                            logger.report_scalar(
                                "train_nelbo_by_kind", KIND_NAMES[k], per_kind[k], step
                            )
                    logger.report_scalar("train", "mean_sub_severity", mean_sev, step)
                kind_loss.zero_()
                kind_count.zero_()
                running_sev = 0.0
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
            if noise_canary is not None:
                nres = noise_canary_eval(
                    eval_model, ema, noise_canary, args.eval_strata, device
                )
                model.train()
                msg = "  ".join(
                    f"{lang} clean {r['clean_ref']:.3f} noised {r['noised']:.3f} "
                    f"null {r['null_slot_bits']:.2f}/letter {r['letter_slot_bits']:.3f}"
                    for lang, r in nres.items()
                )
                print(f"step {step:6d}  heldout noise canary (EMA): {msg}", flush=True)
                if task:
                    for lang, r in nres.items():
                        logger.report_scalar(
                            "heldout_nelbo_noised_bits_per_char",
                            lang,
                            r["noised"],
                            step,
                        )
                        logger.report_scalar(
                            "heldout_nelbo_noised_clean_ref", lang, r["clean_ref"], step
                        )
                        logger.report_scalar(
                            "heldout_null_frame_bits_per_slot",
                            lang,
                            r["framed_per_slot"],
                            step,
                        )
                        logger.report_scalar(
                            "heldout_null_slot_bits", lang, r["null_slot_bits"], step
                        )
                        logger.report_scalar(
                            "heldout_letter_slot_bits",
                            lang,
                            r["letter_slot_bits"],
                            step,
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
            if phase_c:
                # task 4.5 abort criterion: relative degradation vs the start
                # of Phase C, on the EMA and (faster signal) the raw weights.
                degr = {}
                for lang, ref in canary_ref.items():
                    d_ema = cond[lang] / ref - 1.0
                    d_raw = (rcond[lang] / ref - 1.0) if not args.no_eval_raw else d_ema
                    degr[lang] = max(d_ema, d_raw)
                worst = max(degr, key=degr.get)
                print(
                    f"step {step:6d}  canary vs Phase-C start: "
                    + "  ".join(f"{l} {d:+.2%}" for l, d in degr.items()),
                    flush=True,
                )
                if task:
                    for lang, d in degr.items():
                        logger.report_scalar(
                            "phase_c_canary_degradation", lang, d, step
                        )
                if degr[worst] > args.canary_degrade and lambda_max > 0:
                    lambda_max *= 0.5
                    lambda_events.append(
                        {
                            "step": step,
                            "reason": "canary",
                            "language": worst,
                            "degradation": degr[worst],
                            "lambda_max": lambda_max,
                        }
                    )
                    print(
                        f"step {step:6d}  canary breach ({worst} {degr[worst]:+.2%} > "
                        f"{args.canary_degrade:.0%}): λ cap halved to {lambda_max:.4f}",
                        flush=True,
                    )
                lid_eval_head = LIDHead(lid_head.cfg).to(device)
                lid_ema.copy_to(lid_eval_head)
                ema.copy_to(eval_model)
                lid_res = {}
                for name, (ids_e, labels_e) in lid_eval_sets.items():
                    probs = predict(
                        eval_model,
                        lid_eval_head,
                        ids_e,
                        seed=0,
                        device=device,
                        calibrated=False,
                    )
                    lid_res[name] = float((probs.argmax(-1) == labels_e).float().mean())
                model.train()
                print(
                    f"step {step:6d}  heldout LID acc (EMA): "
                    + "  ".join(f"{k} {v:.3f}" for k, v in lid_res.items()),
                    flush=True,
                )
                if task:
                    for k, v in lid_res.items():
                        logger.report_scalar("lid_heldout_acc", k, v, step)
            t_last = time.time()

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=opt,
                scheduler=sched,
                ema=ema,
                step=step,
                extra=ckpt_extra(),
                lid_head=lid_head,
                lid_ema=lid_ema,
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
        extra=ckpt_extra(),
        lid_head=lid_head,
        lid_ema=lid_ema,
    )
    if phase_c:
        print(
            f"λ events: {lambda_events or 'none'}; final λ cap {lambda_max}; "
            f"schedule in {run_dir / 'lambda_schedule.json'}"
        )
    print(f"done at step {step}; checkpoints in {run_dir}")
    if task:
        task.get_logger().flush(wait=True)
        print(f"ClearML: {task.get_output_log_web_page()}")
        task.close()


if __name__ == "__main__":
    main()
