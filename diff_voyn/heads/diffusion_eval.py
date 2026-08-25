"""``DiffusionEvaluator`` — task 5.1 (design §8), the frozen outer tier.

Wraps a *frozen* backbone object behind the same :class:`Evaluator` contract
as the n-gram scorer. Built at CH.4 against random-init weights (interface,
gradient flow through the embedding table, the 2N-slot NULL blend and the
NaN smoke test); Phase 5 loads the Gate-G4 EMA weights through
:meth:`DiffusionEvaluator.from_checkpoint` and nothing else changes
(prototyping doc §9).

Two scoring paths, one instrument:

- **soft / frame path** (:meth:`score_frame`, :meth:`score_fixed`,
  :meth:`score_segmental`): Rao-Blackwellized masked-diffusion NELBO with
  *soft* inputs (expected embeddings) and soft targets, differentiable
  w.r.t. the frame — the dense-gradient refinement signal (R3). Frames
  longer than the backbone context are scored in ≤ ``window`` slot chunks
  and summed. Masking realizations are a pure function of ``seed`` (common
  random numbers across language conditions, non-negotiable #4).
- **hard path** (:meth:`score_ids`): bits/char of hard letter streams via
  the Phase-3 metrology estimator (``metrology.score_conditions``, 64
  stratified draws, CRN across conditions) — the scale every G3/G4 number
  sits on, used for shortlist decisions and the uniform cross-head scale.

Returned soft scores are log-likelihood-scale scalars (−NELBO nats), so
"higher is better" matches the n-gram evaluator.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..data.loader import LANG_TO_INDEX
from ..vocab import LETTER_IDS, MASK_ID, NULL_ID, VOCAB_SIZE
from .evaluator import EvaluatorBase, TokenEmission
from .frame import build_frame, letters_to_vocab
from .ngram import A
from .two_tier import condition_index

LETTER_BASE = LETTER_IDS[0]


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
        window: int = 1024,
        autocast: bool = True,
        meta: dict | None = None,
        stratum_batch: int = 16,
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
        self.window = window
        self.stratum_batch = stratum_batch
        self.autocast = autocast and self.device.type == "cuda"
        self.calibration_offsets_bits = dict(calibration_offsets_bits or {})
        self.calibration_version = calibration_version
        self.meta = dict(meta or {})

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        device: str | torch.device = "cuda",
        calibration_version: str | None = None,
        ema: bool = True,
        **kwargs,
    ) -> DiffusionEvaluator:
        """The Phase-5 swap: the frozen EMA backbone of a training checkpoint
        (``load_backbone`` ignores the LID-head keys of a joint Phase-C
        checkpoint). ``calibration_version`` defaults to the adopted
        ``CALIBRATION_VERSION``."""
        from ..infra.checkpoint import load_backbone
        from ..metrology.calibration import CALIBRATION_VERSION

        if calibration_version is None:
            calibration_version = CALIBRATION_VERSION
        model, meta = load_backbone(Path(path), str(device), ema=ema)
        ev = cls(
            model,
            device=device,
            calibration_version=calibration_version,
            meta=meta,
            **kwargs,
        )
        ev.meta["calibration_version"] = calibration_version
        return ev

    # -- frame scoring core --------------------------------------------------

    def _windows(self, n: int) -> list[tuple[int, int]]:
        """Split ``n`` slots into ≤ ``window`` chunks (last chunk may be
        short; a 2N frame is cut at even offsets so slot pairs stay intact)."""
        if n <= self.window:
            return [(0, n)]
        step = self.window - (self.window % 2)
        return [(s, min(s + step, n)) for s in range(0, n, step)]

    def score_frame(
        self,
        frame: torch.Tensor,
        *,
        language: str,
        seed: int | None = None,
        n_strata: int | None = None,
        letter_slots_only: bool = False,
    ) -> torch.Tensor:
        """(S, VOCAB_SIZE) row-stochastic frame -> −NELBO (nats, scalar),
        differentiable w.r.t. the frame.

        Each stratum masks slots i.i.d. at rate t (replacing their expected
        embedding with the MASK embedding) and pays (1/t)·CE between the
        model's SUBS logits and the frame's own soft target on masked slots.
        ``letter_slots_only`` drops the loss terms of slots whose target is
        (mostly) NULL — the per-plaintext-character accounting of the
        uniform scale (the NULL slots still shape the context).
        """
        frame = frame.to(self.device)
        seed = self.seed if seed is None else seed
        n_strata = self.n_strata if n_strata is None else n_strata
        total = frame.new_zeros(())
        for wi, (a, b) in enumerate(self._windows(frame.shape[0])):
            total = total + self._score_window(
                frame[a:b],
                language,
                seed + 7919 * wi,
                n_strata,
                letter_slots_only,
            )
        return total

    def _score_window(self, frame, language, seed, n_strata, letter_slots_only):
        """All strata of one window in one (or a few) batched forward passes.
        The draw order (per stratum: u, then the S mask draws) matches the
        metrology estimator, so a one-hot frame reproduces its number."""
        S = frame.shape[0]
        g = torch.Generator().manual_seed(seed)
        ts, masks = [], []
        for s in range(n_strata):
            u = torch.rand(1, generator=g).item()
            ts.append(max((s + u) / n_strata, self.t_floor))
            masks.append(torch.rand(S, generator=g) < ts[-1])
        masks = torch.stack(masks).to(self.device)  # (n_strata, S)
        t_vec = torch.tensor(ts, device=self.device, dtype=torch.float32)
        mask_onehot = torch.zeros(VOCAB_SIZE, device=self.device)
        mask_onehot[MASK_ID] = 1.0
        keep = None
        if letter_slots_only:
            keep = frame[:, NULL_ID].detach() < 0.5
        total = frame.new_zeros(())
        for a in range(0, n_strata, self.stratum_batch):
            m = masks[a : a + self.stratum_batch]
            if not m.any():
                continue
            B = m.shape[0]
            z = torch.where(m[:, :, None], mask_onehot[None, None, :], frame[None])
            lang = torch.full((B,), condition_index(language), device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                logits = self.backbone.forward_soft(z, lang)
            logq = torch.log_softmax(logits.float(), dim=-1)  # (B, S, V)
            sel = m if keep is None else (m & keep[None, :])
            # the task-5.1 trap: fill lq with 0 at zero-mass target entries
            # BEFORE the product (0 * -inf = NaN, also in backward)
            f = frame[None].expand(B, S, VOCAB_SIZE)
            lq = logq.masked_fill(f <= 0, 0.0)
            ce = -(f * lq).sum(-1)  # (B, S)
            ce = (ce * sel).sum(-1)  # (B,)
            total = total + (ce / t_vec[a : a + B]).sum() / n_strata
        return -total

    # -- Evaluator contract --------------------------------------------------

    def score_fixed(
        self, soft_letters: torch.Tensor, *, language: str, **kw
    ) -> torch.Tensor:
        return self.score_frame(letters_to_vocab(soft_letters), language=language)

    def score_segmental(
        self, emissions: list[TokenEmission], *, language: str
    ) -> torch.Tensor:
        """Token emissions -> frame -> −NELBO. Convenience fields only
        (``uni``/``pre``/``suf``/``log_w_uni``); rung 3's explicit branch
        lists are collapsed by :func:`emissions_to_frame`."""
        return self.score_frame(emissions_to_frame(emissions), language=language)

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

    # -- hard path: metrology estimator on letter streams ---------------------

    @torch.no_grad()
    def score_ids(
        self,
        letters,
        *,
        language: str | None = None,
        conditions=None,
        n_strata: int = 64,
        seed: int | None = None,
        batch: int = 16,
    ) -> np.ndarray:
        """Bits/char of hard letter-index streams (0..A-1) under one or more
        conditions, via the Phase-3 estimator (stratified draws, CRN across
        conditions within a chunk). ``letters`` is ``[N, L]`` (shared text)
        or ``{condition: [N, L]}`` (one text per condition); returns
        ``[N, C]`` with ``C = len(conditions)`` (a single ``language`` ⇒
        ``C = 1``). Streams longer than ``window`` are tiled into windows and
        averaged per stream (length-weighted)."""
        from ..metrology.scoring import ScoreSettings, score_conditions

        conds = list(conditions) if conditions is not None else [language]
        if any(c is None for c in conds):
            raise ValueError("give language or conditions")
        seed = self.seed if seed is None else seed
        st = ScoreSettings(n_strata=n_strata, seed=seed, batch=batch)
        if isinstance(letters, dict):
            per = {c: np.asarray(letters[c], dtype=np.int64) for c in conds}
        else:
            arr = np.asarray(letters, dtype=np.int64)
            per = {c: arr for c in conds}
        n, L = next(iter(per.values())).shape
        if L <= self.window:
            ids = {c: per[c] + LETTER_BASE for c in conds}
            return score_conditions(
                self.backbone, ids, conds, settings=st, device=self.device
            )
        # tile long streams: equal-length windows (drop the ragged tail only
        # if it is shorter than half a window; otherwise it becomes a window
        # scored on its own chunk)
        out = np.zeros((n, len(conds)))
        for i in range(n):
            cuts = self._windows(L)
            rows = {c: [] for c in conds}
            lens = []
            for a, b in cuts:
                if b - a < self.window // 2 and len(cuts) > 1:
                    continue
                lens.append(b - a)
                for c in conds:
                    rows[c].append(per[c][i, a:b])
            w = np.array(lens, float)
            vals = np.zeros((len(lens), len(conds)))
            for k in range(len(lens)):
                ids = {c: rows[c][k][None] + LETTER_BASE for c in conds}
                stk = ScoreSettings(n_strata=n_strata, seed=seed + 7919 * k, batch=1)
                vals[k] = score_conditions(
                    self.backbone, ids, conds, settings=stk, device=self.device
                )[0]
            out[i] = (vals * w[:, None]).sum(0) / w.sum()
        return out

    def score_stream(
        self, letters: np.ndarray, *, language: str, n_strata: int = 64, seed=None
    ) -> float:
        """Bits/char of one hard stream under one condition."""
        arr = np.asarray(letters, dtype=np.int64)[None]
        return float(
            self.score_ids(arr, language=language, n_strata=n_strata, seed=seed)[0, 0]
        )


def emissions_to_frame(emissions: list[TokenEmission]) -> torch.Tensor:
    """Collapse token emissions to the 2N-slot frame. Explicit branch lists
    (rung 3) are mixed by their (normalized) branch weights: 1-letter
    branches put their distribution in slot 1 and NULL in slot 2; 2-letter
    branches fill both slots. Mixture in probability space (the §8 guard)."""
    s1, s2, w_uni = [], [], []
    for e in emissions:
        branches = list(e.iter_branches())
        logw = torch.stack(
            [
                (w if torch.is_tensor(w) else torch.tensor(float(w))).float()
                for w, _ in branches
            ]
        )
        finite = torch.isfinite(logw)
        if not finite.any():
            raise ValueError("token with no feasible branch (all -inf)")
        pw = torch.softmax(logw.masked_fill(~finite, -1e30), dim=0)
        a = torch.zeros(A)
        b = torch.zeros(A)
        p_uni = torch.zeros(())
        for p, (_, dists) in zip(pw, branches):
            if not torch.isfinite(p) or float(p) == 0.0:
                continue
            a = a + p * dists[0]
            if len(dists) == 1:
                p_uni = p_uni + p
            else:
                b = b + p * dists[1]
        # slot-2 letter mass is (1 - p_uni) * normalized second-letter dist
        if float(1.0 - p_uni) > 0:
            b = b / (1.0 - p_uni)
        else:
            b = torch.full((A,), 1.0 / A)
        s1.append(a)
        s2.append(b)
        w_uni.append(p_uni)
    return build_frame(torch.stack(s1), torch.stack(s2), torch.stack(w_uni))
