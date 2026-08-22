"""Abstain-class data and the language-ID example stream — task 4.3 (design
§6) and the data side of tasks 4.2 / 4.4.

The head's fourth class ("no-language / synthetic") is trained on two
negative controls:

- **voynichesque** — output of the pinned ``voynichesque.py`` (greshko/
  naibbe-cipher @ df3d074, wrapped by :class:`diff_voyn.ciphers.controls.
  Voynichesque`): Voynich-like structured gibberish with *no recoverable
  plaintext* (random alphabets, random re-tokenization, random null
  insertion). Its EVA-style glyph stream is ordinary Latin letters, so after
  the shared normalizer it is an in-vocabulary text like any other — which
  is the point: it must be rejected on structure, not on alphabet. A pool of
  independent encryptions is generated once from *train-split* windows
  (the plaintext content is destroyed by construction; held-out windows feed
  the evaluation pool) and cached under ``DATA_ROOT/abstain/``. The
  generator's parameter sampler is infeasible for ~20% of seeds
  ("not enough options"); those seeds are skipped and the skip count
  recorded.
- **shuffled** — a language window with its letters randomly permuted:
  the unigram statistics of a real language with no sequential structure.
- **uniform** — i.i.d. uniform letters (a small share, Phase-4 addition):
  the first Phase-B head, trained on the two design controls only, labelled
  uniform noise as a language 100% of the time (``lid_eval`` control
  ``uniform_random``, 0% abstain) — both design controls carry unigram
  structure, so "no structure at all" was out of distribution. The share is
  small so the class stays what design §6 intends (structured non-language).

:class:`LIDExampleStream` yields whole batches (one window length per
batch, drawn from ``lengths``) of labelled examples: language windows as
clean / noised / NULL-framed text (the Phase-B :class:`~diff_voyn.data.noise.
NoiseMixture` — "the head sees the same corruption distribution as
deployment", design §6) labelled with their language, and abstain windows
labelled :data:`~diff_voyn.model.lid_head.ABSTAIN_CLASS`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..normalize import normalize
from ..vocab import LETTERS, TOKEN_TO_ID
from .loader import LANG_TO_INDEX, CorpusWindows, LanguageSampler
from .noise import KIND_CLEAN, KIND_NAMES, NoiseConfig, NoiseMixture

ABSTAIN_VERSION = "v1"
KIND_VOYNICHESQUE, KIND_SHUFFLED, KIND_UNIFORM = 4, 5, 6
LID_KIND_NAMES = {
    **KIND_NAMES,
    KIND_VOYNICHESQUE: "voynichesque",
    KIND_SHUFFLED: "shuffled",
    KIND_UNIFORM: "uniform",
}
ABSTAIN_LABEL = len(LANG_TO_INDEX)  # == diff_voyn.model.lid_head.ABSTAIN_CLASS

_LETTER_TABLE = np.full(256, 255, dtype=np.uint8)
for _ch in LETTERS:
    _LETTER_TABLE[ord(_ch)] = TOKEN_TO_ID[_ch]


def encode_normalized(text: str) -> np.ndarray:
    ids = _LETTER_TABLE[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    if (ids == 255).any():
        raise ValueError("non-vocabulary character after normalization")
    return ids


def generate_voynichesque_pool(
    windows: CorpusWindows,
    n_encryptions: int,
    source_chars: int,
    seed: int,
    min_length: int,
) -> tuple[list[np.ndarray], dict]:
    """``n_encryptions`` independent Voynichesque encryptions of random
    ``source_chars``-long windows (languages in the τ-balanced proportion of
    ``windows``), each kept only if it is at least ``min_length`` ids long.
    Deterministic in ``seed``."""
    from ..ciphers.controls import Voynichesque
    from ..vocab import ID_TO_TOKEN

    gen = Voynichesque()
    sampler = LanguageSampler(windows.chars, 0.7)
    rng = np.random.default_rng([seed, 4_3])
    pool: list[np.ndarray] = []
    skipped, too_short, attempt = 0, 0, 0
    by_lang = {l: 0 for l in LANG_TO_INDEX}
    while len(pool) < n_encryptions:
        lang = sampler.sample(rng)
        src = windows.sample_window(lang, source_chars, rng)
        text = "".join(ID_TO_TOKEN[int(i)] for i in src)
        gen_seed = seed * 1_000_003 + attempt
        attempt += 1
        try:
            out = gen.generate(text, seed=gen_seed)
        except ValueError:  # infeasible parameter draw (upstream sampler)
            skipped += 1
            continue
        ids = encode_normalized(normalize(out))
        if len(ids) < min_length:
            too_short += 1
            continue
        pool.append(ids)
        by_lang[lang] += 1
    info = {
        "n_encryptions": len(pool),
        "attempts": attempt,
        "skipped_infeasible": skipped,
        "skipped_too_short": too_short,
        "source_chars": source_chars,
        "source_language_counts": by_lang,
        "min_length": min_length,
        "seed": seed,
        "total_chars": int(sum(len(p) for p in pool)),
    }
    return pool, info


def voynichesque_pool_path(root: Path, split: str) -> Path:
    return root / "abstain" / f"voynichesque_{ABSTAIN_VERSION}_{split}.npz"


def load_or_build_voynichesque_pool(
    root: Path,
    windows: CorpusWindows,
    split: str,
    *,
    n_encryptions: int = 400,
    source_chars: int = 600,
    seed: int = 0,
    min_length: int = 1024,
) -> list[np.ndarray]:
    """Cached pool (``DATA_ROOT/abstain/voynichesque_v1_<split>.npz``) — built
    on first use; the build record sits next to it as ``.json``."""
    path = voynichesque_pool_path(root, split)
    if path.exists():
        z = np.load(path)
        offsets = z["offsets"]
        flat = z["ids"]
        return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    pool, info = generate_voynichesque_pool(
        windows, n_encryptions, source_chars, seed, min_length
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in pool])]).astype(np.int64)
    np.savez(path, ids=np.concatenate(pool), offsets=offsets)
    info["split"] = split
    info["version"] = ABSTAIN_VERSION
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return pool


def sample_from_pool(
    pool: list[np.ndarray], length: int, rng: np.random.Generator
) -> np.ndarray:
    """A random ``length``-window inside one encryption (never across two)."""
    lengths = np.array([len(p) for p in pool], float)
    ok = np.flatnonzero(lengths >= length)
    if len(ok) == 0:
        raise ValueError(f"no pooled encryption reaches {length} chars")
    a = pool[int(rng.choice(ok))]
    start = int(rng.integers(0, len(a) - length + 1))
    return a[start : start + length].copy()


def shuffled_window(ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(ids)


def uniform_random_letters(length: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform i.i.d. letters — the third abstain source (small share) and an
    evaluation control."""
    letter_ids = np.array([TOKEN_TO_ID[c] for c in LETTERS], dtype=np.uint8)
    return rng.choice(letter_ids, size=length)


@dataclass(frozen=True)
class LIDDataConfig:
    """Mix of the head's training stream (tasks 4.2 / 4.3)."""

    p_abstain: float = 0.25
    # shares *within* the abstain examples (sum to 1)
    p_voynichesque: float = 0.45
    p_shuffled: float = 0.40
    p_uniform: float = 0.15
    lengths: tuple[int, ...] = (128, 256, 512, 1024)
    length_probs: tuple[float, ...] = (0.2, 0.25, 0.25, 0.3)
    batch: int = 32
    noise: NoiseConfig | None = None  # None → NoiseConfig() defaults

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_abstain < 1.0:
            raise ValueError("p_abstain must be in [0, 1)")
        if abs(self.p_voynichesque + self.p_shuffled + self.p_uniform - 1.0) > 1e-9:
            raise ValueError("abstain shares must sum to 1")
        if len(self.lengths) != len(self.length_probs):
            raise ValueError("lengths and length_probs must align")
        if abs(sum(self.length_probs) - 1.0) > 1e-9:
            raise ValueError("length_probs must sum to 1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["noise"] = (self.noise or NoiseConfig()).to_dict()
        return d


class LIDExampleStream(torch.utils.data.IterableDataset):
    """Infinite stream of *batches* ``{ids [B, L] int64, label [B], kind [B],
    length}`` (use ``DataLoader(batch_size=None)``). Language examples are
    drawn with the τ-balanced language sampler and corrupted by the Phase-B
    noise mixture (clean / noised / framed / framed+noised kinds); abstain
    examples come from the voynichesque pool or are shuffled language
    windows."""

    def __init__(
        self,
        windows: CorpusWindows,
        pool: list[np.ndarray],
        cfg: LIDDataConfig | None = None,
        seed: int = 0,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.windows = windows
        self.pool = pool
        self.cfg = cfg or LIDDataConfig()
        self.seed = seed
        self.sampler = LanguageSampler(windows.chars, temperature)
        self.mixture = NoiseMixture(self.cfg.noise or NoiseConfig())

    def sample_example(
        self, length: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, int, int]:
        """``(ids[length], label, kind)``."""
        cfg = self.cfg
        if rng.random() < cfg.p_abstain:
            u = rng.random()
            if u < cfg.p_voynichesque:
                return (
                    sample_from_pool(self.pool, length, rng),
                    ABSTAIN_LABEL,
                    KIND_VOYNICHESQUE,
                )
            if u < cfg.p_voynichesque + cfg.p_shuffled:
                lang = self.sampler.sample(rng)
                src = self.windows.sample_window(lang, length, rng)
                return shuffled_window(src, rng), ABSTAIN_LABEL, KIND_SHUFFLED
            return uniform_random_letters(length, rng), ABSTAIN_LABEL, KIND_UNIFORM
        lang = self.sampler.sample(rng)
        kind = self.mixture.sample_kind(rng)
        src = self.windows.sample_window(
            lang, self.mixture.source_length(kind, length), rng
        )
        ids, _ = self.mixture.apply(src, kind, length, rng)
        return ids, LANG_TO_INDEX[lang], kind

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, 4, wid])
        lengths = np.array(self.cfg.lengths)
        probs = np.array(self.cfg.length_probs)
        while True:
            length = int(rng.choice(lengths, p=probs))
            ids, labels, kinds = [], [], []
            for _ in range(self.cfg.batch):
                x, y, k = self.sample_example(length, rng)
                ids.append(x)
                labels.append(y)
                kinds.append(k)
            yield {
                "ids": torch.from_numpy(np.stack(ids).astype(np.int64)),
                "label": torch.tensor(labels, dtype=torch.long),
                "kind": torch.tensor(kinds, dtype=torch.long),
                "length": length,
            }


def build_lid_eval_set(
    heldout: CorpusWindows,
    pool_heldout: list[np.ndarray],
    *,
    n_per_language: int = 32,
    lengths: tuple[int, ...] = (128, 256, 1024),
    seed: int = 54321,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Fixed held-out LID sets for the training canary: per length, clean
    language windows (label = language), the moderate fixed-severity noised
    variant, the NULL-framed variant, voynichesque and shuffled windows
    (label = abstain). Deterministic in ``seed``."""
    from .noise import framed_variant, noised_variant

    rng = np.random.default_rng(seed)
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    src_margin = NoiseConfig().source_margin
    for L in lengths:
        clean, noised, framed, labels = [], [], [], []
        for lang, li in LANG_TO_INDEX.items():
            for _ in range(n_per_language):
                src = heldout.sample_window(lang, int(L * src_margin), rng)
                clean.append(src[:L])
                noised.append(noised_variant(src, L, rng))
                framed.append(framed_variant(src, L, rng))
                labels.append(li)
        n_abs = n_per_language * len(LANG_TO_INDEX)
        voy = [sample_from_pool(pool_heldout, L, rng) for _ in range(n_abs)]
        uni = [uniform_random_letters(L, rng) for _ in range(n_abs)]
        shuf = [
            shuffled_window(
                heldout.sample_window(
                    list(LANG_TO_INDEX)[i % len(LANG_TO_INDEX)], L, rng
                ),
                rng,
            )
            for i in range(n_abs)
        ]
        lab = torch.tensor(labels, dtype=torch.long)
        abs_lab = torch.full((n_abs,), ABSTAIN_LABEL, dtype=torch.long)
        for name, arrs, y in (
            ("clean", clean, lab),
            ("noised", noised, lab),
            ("framed", framed, lab),
            ("voynichesque", voy, abs_lab),
            ("shuffled", shuf, abs_lab),
            ("uniform", uni, abs_lab),
        ):
            out[f"{name}_L{L}"] = (
                torch.from_numpy(np.stack(arrs).astype(np.int64)),
                y,
            )
    return out


def kind_name(kind: int) -> str:
    return LID_KIND_NAMES.get(int(kind), str(kind))


def is_clean_kind(kind: int) -> bool:
    return int(kind) == KIND_CLEAN


__all__ = [
    "ABSTAIN_LABEL",
    "ABSTAIN_VERSION",
    "KIND_SHUFFLED",
    "KIND_UNIFORM",
    "KIND_VOYNICHESQUE",
    "LID_KIND_NAMES",
    "LIDDataConfig",
    "LIDExampleStream",
    "build_lid_eval_set",
    "encode_normalized",
    "generate_voynichesque_pool",
    "load_or_build_voynichesque_pool",
    "sample_from_pool",
    "shuffled_window",
    "uniform_random_letters",
]
