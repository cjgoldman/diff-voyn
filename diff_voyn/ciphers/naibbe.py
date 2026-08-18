"""Naibbe cipher wrapper — task 0.7 (design §9, pinned to naibbe_v2 @ df3d074).

The upstream module reads ``references/naibbe_tables.csv`` relative to its own
repo and draws from the stdlib ``random`` module, so this wrapper imports it
with the right working directory and seeds through the module's RNG.

Preprocessing split (task 0.1 note):

- *Model-side*: our shared normalizer (``diff_voyn.normalize``) — j→i etc.
- *Cipher-side*: the Naibbe cipher is defined over 23 letters with no k/w;
  its own lossy pre-map (k→c, w→uu, from upstream ``clean_line``) is applied
  here, inside the cipher, never in the shared pipeline. The plaintext half of
  every generated pair is returned in cipher-alphabet space so pairs stay
  aligned.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys

from ..normalize import normalize
from ..vocab import NAIBBE_LETTERS
from .external import naibbe_repo

_CIPHER_SIDE_MAP = str.maketrans({"k": "c"})  # + w→uu below (1→2 chars)


def naibbe_pre_map(normalized_text: str) -> str:
    """Map our 25-letter normalized stream onto Naibbe's 23-letter alphabet."""
    out = normalized_text.replace("w", "uu").translate(_CIPHER_SIDE_MAP)
    assert set(out) <= set(NAIBBE_LETTERS)
    return out


class NaibbeCipher:
    """Seedable wrapper around the pinned ``naibbe_v2`` implementation."""

    def __init__(self, seed: int = 0, use_78_card_deck: bool = False):
        repo = naibbe_repo()
        cwd = os.getcwd()
        try:
            os.chdir(repo)  # upstream reads references/naibbe_tables.csv via cwd
            spec = importlib.util.spec_from_file_location(
                "naibbe_v2", repo / "naibbe_v2.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["naibbe_v2"] = mod
            spec.loader.exec_module(mod)
        finally:
            os.chdir(cwd)
        self._mod = mod
        self.use_78 = use_78_card_deck
        self.seed(seed)

    def seed(self, seed: int) -> None:
        self._mod.random.seed(seed)

    def encipher(self, text: str) -> tuple[list[str], list[str]]:
        """Encipher ``text`` (any raw string; normalized internally).

        Returns ``(cipher_tokens, plaintext_segments)`` where segment i (one
        or two letters, cipher-alphabet space) produced token i — the aligned
        pair needed for paired corpora and for rung-3 ground truth (task 5.4).
        """
        plain = naibbe_pre_map(normalize(text))
        seg_capture = io.StringIO()
        tokens = self._mod.encrypt_naibbe(
            plain,
            self._mod.naibbe_tables,
            self._mod.placeholder_to_glyph,
            use_78=self.use_78,
            pre_plaintext_file=seg_capture,
        )
        segments = seg_capture.getvalue().split()
        assert "".join(segments) == plain
        assert len(segments) == len(tokens)
        return tokens, segments

    def ciphertext_stream(self, text: str) -> str:
        """Whitespace-free ciphertext characters, as the model pipeline sees it
        (design §2: Naibbe's re-spacing layer is absorbed by whitespace removal)."""
        tokens, _ = self.encipher(text)
        return "".join(tokens)
