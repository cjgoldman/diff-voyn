"""Exponential moving average of model weights (decay 0.9999, design §7.5).

The EMA weights are what every evaluation uses, and — frozen — they are the
Phase-5 cryptanalysis evaluator.
"""

from __future__ import annotations

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            v.copy_(self.shadow[k].to(v.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]
