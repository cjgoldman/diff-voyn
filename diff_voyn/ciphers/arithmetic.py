"""Arithmetic (pseudo-VMS sum-to-target) cipher wrapper — task 0.7, design §10.

Pinned to ``voynpy.pseudo_vms.PseudoVmsEncoder`` at the SHA in
``external.py``. Defaults are upstream defaults (zipf_exponent, tokens_per_char,
length distribution, seed=42); the alphabet is our frozen 25-letter vocab with
each letter keeping its upstream default value (a=3 … z=28, j's slot unused),
so statistics stay comparable with upstream experiments.

``doubling_strength`` is tuned per language via ``tune_to_vms`` against the
VMS ~0.92% token-doubling rate (a per-source-language calibration, design
§10); tuned tables are persisted with ``save()`` for persisted-determinism.
"""

from __future__ import annotations

from pathlib import Path

from ..vocab import LETTERS
from .external import ensure_voynpy_importable

VMS_DOUBLING_RATE = 0.0092


def our_alphabet_values() -> dict[str, int]:
    """Frozen letters → upstream default integer values (a=3 … z=28)."""
    return {c: 3 + (ord(c) - ord("a")) for c in LETTERS}


class ArithmeticCipher:
    def __init__(self, table_path: Path | None = None, **overrides):
        ensure_voynpy_importable()
        from voynpy.pseudo_vms import PseudoVmsEncoder

        if table_path is not None and Path(table_path).exists():
            self.enc = PseudoVmsEncoder.load(table_path, **overrides)
        else:
            self.enc = PseudoVmsEncoder(alphabet=our_alphabet_values(), **overrides)

    def tune_doubling(
        self, sample_text: str, target: float = VMS_DOUBLING_RATE
    ) -> float:
        return self.enc.tune_to_vms(
            target_doubling_rate=target, sample_text=sample_text
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.enc.save(path)

    def encode(self, normalized_text: str) -> str:
        return self.enc.encode(normalized_text)

    def decode_text(self, cipher: str) -> str:
        return self.enc.decode_text(cipher)

    def stats(self, cipher: str) -> dict:
        """Entropy / token-length distribution / doubling rate of a cipher
        stream (the acceptance-logging numbers for task 0.7)."""
        import math
        from collections import Counter
        from itertools import pairwise

        tokens = cipher.split()
        chars = cipher.replace(" ", "").replace("\n", "")
        counts = Counter(chars)
        n = sum(counts.values())
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        lengths = Counter(len(t) for t in tokens)
        doubles = sum(1 for a, b in pairwise(tokens) if a == b)
        return {
            "tokens": len(tokens),
            "char_entropy_bits": entropy,
            "token_length_dist": {
                k: v / len(tokens) for k, v in sorted(lengths.items())
            },
            "doubling_rate": doubles / max(len(tokens) - 1, 1),
            "doubling_strength": self.enc.doubling_strength,
        }
