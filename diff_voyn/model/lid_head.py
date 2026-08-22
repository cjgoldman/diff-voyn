"""Language-ID head — task 4.1 (design §6, §7.2).

Architecture (design §6): mean-pool the backbone's final hidden states
(:meth:`diff_voyn.model.backbone.Backbone.hidden`), computed at several
masking levels and averaged → 2-layer MLP → softmax over the frozen language
inventory plus one **abstain** class ("no-language / synthetic", trained on
``voynichesque.py`` output and shuffled text, task 4.3).

Decisions fixed here:

- **The feature pass is unconditional.** Features are taken under the
  NULL-language embedding (:data:`~diff_voyn.data.loader.NULL_LANG_INDEX`),
  never under a language condition: a head reading features computed under
  the true language's conditioning embedding would learn the label from the
  conditioning signal, not from the text (and at deployment the hypothesis
  is exactly what is unknown). The head classifies *text*; the per-language
  ELBO — the primary metric — classifies *(text, condition)* pairs.
- **Masking levels** ``LID_MASK_LEVELS``: the input is masked at each level
  with common random numbers (one generator, design §5a) and the pooled
  features are averaged. The same levels are used in training and at
  inference, so there is no train/test feature shift. Level 0 (unmasked
  text) is included: it carries the most evidence and the backbone sees
  near-unmasked inputs throughout training (t → 0).
- **Stop-gradient switch** (``stop_gradient=True`` in Phase B, task 4.2):
  features are computed under ``torch.no_grad`` and detached, so backbone
  gradients are *exactly* zero (tested). Phase C (task 4.4) releases it and
  the LID loss is weighted by λ (:func:`lambda_schedule`).
- **Temperature** (task 4.6): ``log_temperature`` is a buffer, zero until
  :mod:`scripts.head_calibration` fits it on held-out decipherments;
  :meth:`LIDHead.calibrated_logits` divides by ``exp(log_temperature)``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..data.loader import LANG_TO_INDEX, NULL_LANG_INDEX
from ..vocab import MASK_ID

LID_CLASSES: tuple[str, ...] = tuple(LANG_TO_INDEX) + ("abstain",)
ABSTAIN_CLASS: int = len(LANG_TO_INDEX)  # 3
N_LID_CLASSES: int = len(LID_CLASSES)
LID_CLASS_INDEX: dict[str, int] = {c: i for i, c in enumerate(LID_CLASSES)}
assert all(LID_CLASS_INDEX[l] == i for l, i in LANG_TO_INDEX.items())

# Masking levels whose pooled features are averaged (design §6: "computed at
# several masking levels, averaged").
LID_MASK_LEVELS: tuple[float, ...] = (0.0, 0.15, 0.3, 0.5)


@dataclass(frozen=True)
class LIDHeadConfig:
    d_model: int = 512
    hidden: int = 512
    n_classes: int = N_LID_CLASSES
    dropout: float = 0.1
    mask_levels: tuple[float, ...] = LID_MASK_LEVELS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mask_levels"] = list(self.mask_levels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LIDHeadConfig:
        d = dict(d)
        d["mask_levels"] = tuple(float(x) for x in d["mask_levels"])
        return cls(**d)


class LIDHead(nn.Module):
    """``forward(features [B, d_model]) -> logits [B, n_classes]``."""

    def __init__(self, cfg: LIDHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.hidden, cfg.n_classes)
        self.register_buffer("log_temperature", torch.zeros(()))
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.norm(features.float()))
        return self.fc2(self.dropout(F.gelu(h)))

    def calibrated_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Logits divided by the fitted temperature (task 4.6)."""
        return self.forward(features) / self.log_temperature.exp()

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def mask_at_level(
    ids: torch.Tensor, level: float, g: torch.Generator | None = None
) -> torch.Tensor:
    """Replace each position by MASK with probability ``level`` (the forward
    process at a fixed t); ``level == 0`` returns the ids unchanged."""
    if level <= 0.0:
        return ids
    u = torch.rand(ids.shape, generator=g).to(ids.device)
    return ids.masked_fill(u < level, MASK_ID)


def pooled_features(
    backbone: nn.Module,
    ids: torch.Tensor,
    mask_levels: tuple[float, ...] = LID_MASK_LEVELS,
    *,
    g: torch.Generator | None = None,
    stop_gradient: bool = False,
    autocast: bool = False,
) -> torch.Tensor:
    """``[B, d_model]`` — mean over positions of the unconditional final
    hidden states, averaged over ``mask_levels`` (common random numbers
    through ``g``). With ``stop_gradient`` the backbone sees no autograd at
    all (task 4.2: backbone unchanged)."""
    lang = torch.full(
        (ids.shape[0],), NULL_LANG_INDEX, dtype=torch.long, device=ids.device
    )
    ctx = torch.no_grad() if stop_gradient else nullcontext()
    feats = None
    with ctx:
        for level in mask_levels:
            z = mask_at_level(ids, level, g)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=autocast and ids.is_cuda
            ):
                h = backbone.hidden(z, lang)
            pooled = h.float().mean(dim=1)
            feats = pooled if feats is None else feats + pooled
        feats = feats / len(mask_levels)
    return feats.detach() if stop_gradient else feats


def lid_logits(
    backbone: nn.Module,
    head: LIDHead,
    ids: torch.Tensor,
    *,
    g: torch.Generator | None = None,
    stop_gradient: bool = False,
    autocast: bool = False,
    calibrated: bool = False,
) -> torch.Tensor:
    feats = pooled_features(
        backbone,
        ids,
        head.cfg.mask_levels,
        g=g,
        stop_gradient=stop_gradient,
        autocast=autocast,
    )
    return head.calibrated_logits(feats) if calibrated else head(feats)


def lid_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels)


@torch.no_grad()
def predict(
    backbone: nn.Module,
    head: LIDHead,
    ids: torch.Tensor,
    *,
    batch: int = 32,
    seed: int = 0,
    device: str | torch.device = "cpu",
    calibrated: bool = True,
    autocast: bool = True,
) -> torch.Tensor:
    """Class probabilities ``[N, n_classes]`` for ``ids`` [N, L] (eval mode,
    fixed masking seed so repeated calls agree)."""
    backbone.eval()
    head.eval()
    out = []
    for start in range(0, ids.shape[0], batch):
        chunk = ids[start : start + batch].to(device)
        g = torch.Generator().manual_seed(seed + start)
        logits = lid_logits(
            backbone,
            head,
            chunk,
            g=g,
            stop_gradient=True,
            autocast=autocast,
            calibrated=calibrated,
        )
        out.append(F.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out, dim=0)


def lambda_schedule(step: int, ramp_steps: int, lambda_max: float) -> float:
    """Phase-C LID loss weight: linear ramp 0 → ``lambda_max`` over
    ``ramp_steps`` optimizer steps, then flat (design §7.2). ``lambda_max``
    is the *current* cap — the training loop halves it on a canary breach
    (task 4.5) or when the LID gradient exceeds 10% of the diffusion
    gradient (task 4.4)."""
    if ramp_steps <= 0:
        return lambda_max
    return lambda_max * min(1.0, step / ramp_steps)


def backbone_grad_norm(backbone: nn.Module) -> float:
    """L2 norm of the gradients currently held by the backbone parameters."""
    sq = 0.0
    for p in backbone.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return sq**0.5
