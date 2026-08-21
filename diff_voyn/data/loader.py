"""Data loader — task 0.5.

Pieces:

- :class:`LanguageSampler` — temperature-balanced language sampling (τ≈0.7 over
  per-language character counts, design §4). Weights are exposed for logging:
  every run manifest records them (task 0.5 "per-language weights logged").
- :class:`CorpusWindows` — memory-maps the normalized per-document corpus and
  yields random fixed-length character windows per language.
- :class:`MaskingSampler` — the MDLM forward process at a sampled time
  ``t ~ U(0,1)`` under the log-linear schedule (α_t = 1−t): every position is
  independently replaced by MASK with probability t; the Rao-Blackwellized
  NELBO weight for that draw is 1/t (design §1). Supports common random
  numbers (task 3.1) by accepting an explicit generator.
- :class:`LanguageConditioning` — the additive per-position language embedding
  with p=0.1 conditioning dropout to a learned NULL-language embedding
  (design §4). Included here so injection mechanics are testable in Phase 0;
  the Phase-1 model composes it.
- :class:`DiffVoynIterableDataset` — infinite stream of training examples;
  with a :class:`~diff_voyn.data.noise.NoiseMixture` attached (Phase B, tasks
  2.4/2.5) each example is drawn as clean / noised / NULL-framed / both, and
  the example kind is emitted for logging.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..vocab import MASK_ID, TOKEN_TO_ID
from .noise import KIND_CLEAN, NoiseMixture

# Frozen inventory (task 0.2) in frozen index order.
LANG_TO_INDEX: dict[str, int] = {"latin": 0, "italian": 1, "german": 2}
NULL_LANG_INDEX: int = len(LANG_TO_INDEX)  # unconditional / conditioning-dropout
N_LANG_EMBEDDINGS: int = len(LANG_TO_INDEX) + 1


class LanguageSampler:
    """Temperature-smoothed sampling over languages: p_l ∝ n_l^τ."""

    def __init__(self, chars_per_language: dict[str, int], temperature: float = 0.7):
        self.languages: list[str] = sorted(chars_per_language)
        counts = np.array([chars_per_language[l] for l in self.languages], float)
        if (counts <= 0).any():
            raise ValueError("every language needs a positive character count")
        w = counts**temperature
        self.weights: np.ndarray = w / w.sum()
        self.temperature = temperature

    def weights_dict(self) -> dict[str, float]:
        return {l: float(p) for l, p in zip(self.languages, self.weights)}

    def sample(self, rng: np.random.Generator, size: int | None = None):
        idx = rng.choice(len(self.languages), size=size, p=self.weights)
        if size is None:
            return self.languages[int(idx)]
        return [self.languages[i] for i in np.atleast_1d(idx)]


class CorpusWindows:
    """Random fixed-length windows over the normalized, encoded corpus.

    Documents are loaded from ``corpora/<version>/<lang>/docs/<doc_id>.txt``
    (restricted to the doc ids given, i.e. the train side of the splits file),
    encoded once to uint8 id arrays, and sampled proportionally to length.
    """

    def __init__(self, corpus_dir: Path, doc_ids_per_lang: dict[str, Sequence[str]]):
        table = np.full(256, 255, dtype=np.uint8)
        for ch, i in TOKEN_TO_ID.items():
            if len(ch) == 1:
                table[ord(ch)] = i
        self.docs: dict[str, list[np.ndarray]] = {}
        self.doc_ids: dict[str, list[str]] = {}
        self.doc_weights: dict[str, np.ndarray] = {}
        self.chars: dict[str, int] = {}
        for lang, doc_ids in doc_ids_per_lang.items():
            self.doc_ids[lang] = list(doc_ids)
            arrs = []
            for doc_id in doc_ids:
                b = np.frombuffer(
                    (corpus_dir / lang / "docs" / f"{doc_id}.txt").read_bytes(),
                    dtype=np.uint8,
                )
                ids = table[b]
                if (ids == 255).any():
                    raise ValueError(f"non-vocab byte in {lang}/{doc_id}")
                arrs.append(ids)
            if not arrs:
                raise ValueError(f"no documents for language {lang!r}")
            self.docs[lang] = arrs
            lengths = np.array([len(a) for a in arrs], float)
            self.doc_weights[lang] = lengths / lengths.sum()
            self.chars[lang] = int(lengths.sum())

    def sample_window(
        self, lang: str, length: int, rng: np.random.Generator
    ) -> np.ndarray:
        arrs = self.docs[lang]
        a = arrs[int(rng.choice(len(arrs), p=self.doc_weights[lang]))]
        if len(a) <= length:  # short doc: take all, wrap by tiling
            reps = math.ceil((length + 1) / len(a))
            a = np.tile(a, reps)
        start = int(rng.integers(0, len(a) - length + 1))
        return a[start : start + length].copy()

    def tiled_windows(self, lang: str, length: int) -> np.ndarray:
        """Deterministic full coverage: every document cut into consecutive
        non-overlapping ``length``-char windows (tail remainder dropped),
        concatenated in doc order → ``[n_windows, length]`` uint8.

        The evaluation counterpart of :meth:`sample_window`: calibration
        (task 3.4) scores the diffusion NELBO and the AR reference NLL on
        exactly these windows, so the two numbers share their text."""
        return self.tiled_windows_by_doc(lang, length)[0]

    def tiled_windows_by_doc(
        self, lang: str, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`tiled_windows` plus the document index of every window
        (``[n_windows]`` int64, indexing ``self.doc_ids[lang]``) — the key for
        per-document mean-and-spread reporting (task 3.3)."""
        chunks, doc_index = [], []
        for di, a in enumerate(self.docs[lang]):
            n = len(a) // length
            if n:
                chunks.append(a[: n * length].reshape(n, length))
                doc_index.append(np.full(n, di, dtype=np.int64))
        if not chunks:
            raise ValueError(f"no {lang} document reaches {length} chars")
        return np.concatenate(chunks, axis=0).copy(), np.concatenate(doc_index)


class MaskingSampler:
    """Absorbing-state forward process q(z_t | x) at sampled t (log-linear α_t=1−t)."""

    def __init__(self, t_floor: float = 1e-3):
        # t is clamped away from 0: the 1/t NELBO weight diverges and no
        # position would be masked anyway.
        self.t_floor = t_floor

    def sample_t(self, batch: int, g: torch.Generator | None = None) -> torch.Tensor:
        t = torch.rand(batch, generator=g)
        return t.clamp_min(self.t_floor)

    def mask(
        self, ids: torch.Tensor, t: torch.Tensor, g: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_t, mask_positions, nelbo_weight).

        ``nelbo_weight`` is 1/t per example — multiply the mean CE over masked
        positions by it to get the Rao-Blackwellized NELBO integrand.
        """
        if ids.dim() != 2:
            raise ValueError("ids must be [batch, seq]")
        u = torch.rand(ids.shape, generator=g)
        masked = u < t[:, None]
        z_t = ids.masked_fill(masked, MASK_ID)
        return z_t, masked, 1.0 / t


class LanguageConditioning(nn.Module):
    """Additive per-position language embedding with conditioning dropout (§4)."""

    def __init__(self, d_model: int, p_dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(N_LANG_EMBEDDINGS, d_model)
        self.p_dropout = p_dropout
        # Realized drop fraction of the most recent training-mode forward,
        # accumulated by the train loop so logs verify the 10% rate (task 1.2).
        self.last_drop_fraction: float = 0.0

    def forward(
        self, lang_idx: torch.Tensor, g: torch.Generator | None = None
    ) -> torch.Tensor:
        """lang_idx [batch] → additive embedding [batch, 1, d_model].

        In training mode, each example's language index is replaced by
        NULL_LANG_INDEX with probability ``p_dropout`` (the unconditional
        pathway of design §4). Eval mode conditions exactly as asked.
        """
        if self.training and self.p_dropout > 0:
            drop = torch.rand(lang_idx.shape, generator=g, device=lang_idx.device)
            dropped = drop < self.p_dropout
            self.last_drop_fraction = float(dropped.float().mean())
            lang_idx = torch.where(
                dropped,
                torch.full_like(lang_idx, NULL_LANG_INDEX),
                lang_idx,
            )
        return self.embedding(lang_idx)[:, None, :]


class DiffVoynIterableDataset(torch.utils.data.IterableDataset):
    """Infinite stream: {ids, z_t, mask, t, weight, lang_idx, kind, sub_severity}.

    ``kind`` is the example type (``noise.KIND_*``; always 0 = clean without a
    mixture) and ``sub_severity`` the wrong-key severity applied (0 if none) —
    both are consumed only by logging, so the realized Phase-B mix is
    auditable on the dashboard.
    """

    def __init__(
        self,
        windows: CorpusWindows,
        seq_len: int = 1024,
        temperature: float = 0.7,
        seed: int = 0,
        noise: NoiseMixture | None = None,
    ):
        super().__init__()
        self.windows = windows
        self.seq_len = seq_len
        self.sampler = LanguageSampler(windows.chars, temperature)
        self.masking = MaskingSampler()
        self.seed = seed
        self.noise = noise

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        g = torch.Generator().manual_seed(self.seed * 1000 + wid)
        while True:
            lang = self.sampler.sample(rng)
            kind, sub_severity = KIND_CLEAN, 0.0
            if self.noise is None:
                window = self.windows.sample_window(lang, self.seq_len, rng)
            else:
                kind = self.noise.sample_kind(rng)
                source = self.windows.sample_window(
                    lang, self.noise.source_length(kind, self.seq_len), rng
                )
                window, ninfo = self.noise.apply(source, kind, self.seq_len, rng)
                sub_severity = ninfo.get("substitution", {}).get("severity", 0.0)
            ids = torch.from_numpy(window.astype(np.int64))[None, :]
            t = self.masking.sample_t(1, g)
            z_t, masked, weight = self.masking.mask(ids, t, g)
            yield {
                "ids": ids[0],
                "z_t": z_t[0],
                "mask": masked[0],
                "t": t[0],
                "weight": weight[0],
                "lang_idx": torch.tensor(LANG_TO_INDEX[lang]),
                "kind": torch.tensor(kind),
                "sub_severity": torch.tensor(sub_severity, dtype=torch.float32),
            }
