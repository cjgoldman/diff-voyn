"""Checkpoint + resume with full RNG capture (task 0.6).

Acceptance: kill-and-resume reproduces the loss curve — which requires saving
optimizer state, EMA shadow, step counter, and every RNG stream (python,
numpy, torch CPU/CUDA). Verified by ``tests/test_infra.py``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ema import EMA


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMA | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "ema": ema.state_dict() if ema else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a killed save never corrupts the checkpoint


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMA | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None and state["ema"] is not None:
        ema.load_state_dict(state["ema"])
    if restore_rng:
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if rng["torch_cuda"] is not None and torch.cuda.is_available():
            # A checkpoint may have been saved with more GPUs visible than now
            # (e.g. resumed under CUDA_VISIBLE_DEVICES pinning) — restore what
            # exists; spare states are irrelevant to the visible devices.
            n = min(len(rng["torch_cuda"]), torch.cuda.device_count())
            for i in range(n):
                torch.cuda.set_rng_state(rng["torch_cuda"][i], i)
    return state


def load_backbone(path: Path, device: str = "cpu", *, ema: bool = True):
    """Build the backbone recorded in a training checkpoint and load its EMA
    (default) or raw weights. Returns ``(model.eval(), meta)`` where ``meta``
    carries path/step/ema_decay/schedule/model config for reports."""
    from ..infra.config import ModelConfig
    from ..model.backbone import Backbone

    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["extra"]["config"]["model"])
    model = Backbone(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    source = "raw"
    if ema and state.get("ema") is not None:
        sd = model.state_dict()
        for k, v in state["ema"]["shadow"].items():
            sd[k].copy_(v.to(sd[k].dtype))
        source = "ema"
    meta = {
        "path": str(path),
        "step": state["step"],
        "weights": source,
        "ema_decay": state["ema"]["decay"] if state.get("ema") else None,
        "schedule": state["extra"].get("schedule"),
        "model": state["extra"]["config"]["model"],
        "phase": state["extra"]["config"].get("phase"),
        "run_name": state["extra"]["config"].get("run_name"),
    }
    return model, meta
