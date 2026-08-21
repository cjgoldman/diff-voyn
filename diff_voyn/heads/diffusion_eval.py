"""``DiffusionEvaluator`` — CH.4 / task 5.1 plumbing (design §8).

Wraps a *frozen* backbone object behind the same :class:`Evaluator` contract
as the n-gram scorer. Today the weights are random-init (the G0 artifact) and
the scores are meaningless — but the interface, gradient flow through the
embedding table, the 2N-slot NULL blend, and the NaN smoke test are all fully
exercisable now. Post-G4 the EMA weights are loaded and nothing else changes
(prototyping doc §9).

Scoring: Rao-Blackwellized masked-diffusion NELBO with *soft* inputs and
targets. Masking realizations are a pure function of ``seed`` — score every
candidate language with the same seed for common random numbers (design §5a,
non-negotiable #4). Returned score is a log-likelihood-scale scalar
(−NELBO nats), so "higher is better" matches the n-gram evaluator.
"""

from __future__ import annotations

import math

import torch

from ..data.loader import LANG_TO_INDEX
from ..vocab import MASK_ID, VOCAB_SIZE
from .evaluator import EvaluatorBase, TokenEmission
from .frame import build_frame, letters_to_vocab
from .ngram import A


class DiffusionEvaluator(EvaluatorBase):
    """Frozen-backbone plaintext scorer over the 2N-slot frame."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        n_strata: int = 8,
        seed: int = 0,
        t_floor: float = 1e-3,
        device: str | torch.device = "cpu",
        calibration_offsets_bits: dict[str, float] | None = None,
        calibration_version: str | None = None,
    ):
        """``calibration_version`` loads the versioned §3.4 table
        (``metrology.CalibrationTable``) and installs its offsets in the
        hook's additive convention; ``calibration_offsets_bits`` passes
        additive offsets directly (mutually exclusive)."""
        if calibration_version is not None:
            if calibration_offsets_bits is not None:
                raise ValueError("give calibration_version or offsets, not both")
            from ..metrology.calibration import CalibrationTable

            calibration_offsets_bits = CalibrationTable.load(
                calibration_version
            ).additive_offsets()
        self.backbone = backbone.to(device).eval()
        for p in self.backbone.parameters():  # frozen measuring stick (§7.4)
            p.requires_grad_(False)
        self.languages = sorted(LANG_TO_INDEX)
        self.n_strata = n_strata
        self.seed = seed
        self.t_floor = t_floor
        self.device = torch.device(device)
        self.calibration_offsets_bits = dict(calibration_offsets_bits or {})

    # -- frame scoring core --------------------------------------------------

    def score_frame(self, frame: torch.Tensor, *, language: str) -> torch.Tensor:
        """(S, VOCAB_SIZE) row-stochastic frame -> −NELBO (nats, scalar),
        differentiable w.r.t. the frame.

        Each stratum masks slots i.i.d. at rate t (replacing their expected
        embedding with the MASK embedding) and pays (1/t)·CE between the
        model's SUBS logits and the frame's own soft target on masked slots.
        """
        frame = frame.to(self.device)
        S = frame.shape[0]
        lang = torch.tensor([LANG_TO_INDEX[language]], device=self.device)
        g = torch.Generator().manual_seed(self.seed)
        mask_onehot = torch.zeros(VOCAB_SIZE, device=self.device)
        mask_onehot[MASK_ID] = 1.0

        total = frame.new_zeros(())
        for s in range(self.n_strata):
            u = torch.rand(1, generator=g).item()
            t = max((s + u) / self.n_strata, self.t_floor)
            masked = (torch.rand(S, generator=g) < t).to(self.device)
            if not masked.any():
                continue
            z = torch.where(masked[:, None], mask_onehot[None, :], frame)
            logits = self.backbone.forward_soft(z[None], lang)
            logq = torch.log_softmax(logits[0], dim=-1)
            # Soft-target CE on masked slots; frame rows are targets. SUBS
            # sets the MASK logit to -inf and the frame is 0 there, so the
            # naive f*lq forward is 0*-inf = NaN — and a torch.where guard
            # still NaNs in *backward* (the mul node's grad is 0 * -inf).
            # Fill lq with 0 at zero-mass target entries BEFORE the product
            # (the task-5.1 trap; caught by the smoke test).
            f = frame[masked]
            lq = logq[masked].masked_fill(f <= 0, 0.0)
            ce = -(f * lq).sum()
            total = total + (1.0 / t) * ce / self.n_strata
        return -total

    # -- Evaluator contract --------------------------------------------------

    def score_fixed(self, soft_letters: torch.Tensor, *, language: str) -> torch.Tensor:
        return self.score_frame(letters_to_vocab(soft_letters), language=language)

    def score_segmental(
        self, emissions: list[TokenEmission], *, language: str
    ) -> torch.Tensor:
        uni = torch.stack([e.uni if e.uni is not None else e.pre for e in emissions])
        slot1 = torch.stack([e.pre if e.pre is not None else e.uni for e in emissions])
        suf = torch.stack(
            [
                e.suf if e.suf is not None else torch.full((A,), 1.0 / A)
                for e in emissions
            ]
        )
        w_uni = torch.stack(
            [
                torch.exp(torch.as_tensor(e.log_w_uni, dtype=torch.float32))
                for e in emissions
            ]
        ).clamp(0.0, 1.0)
        # slot 1: mixture of unigram letter (weight w) and prefix letter (1-w);
        # slot 2: NULL-blended suffix (build_frame applies the blend).
        s1 = w_uni[:, None] * uni + (1.0 - w_uni)[:, None] * slot1
        frame = build_frame(s1, suf, w_uni)
        return self.score_frame(frame, language=language)

    def as_embedding_frame(
        self, soft_letters: torch.Tensor, null_weights: torch.Tensor
    ) -> torch.Tensor:
        """Letters + per-token NULL weights -> expected embeddings (2N, d)."""
        frame = build_frame(
            soft_letters[0::2], soft_letters[1::2], null_weights.to(soft_letters)
        )
        return frame.to(self.device) @ self.backbone.embed.weight

    def bits_per_slot(self, frame: torch.Tensor, *, language: str) -> float:
        s = self.score_frame(frame, language=language)
        return float(-s) / (frame.shape[0] * math.log(2.0))
