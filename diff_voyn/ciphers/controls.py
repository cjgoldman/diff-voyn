"""Negative control: voynichesque.py (task 0.7; consumed by 4.3 / 6.3).

Voynich-like structured gibberish generated from REAL held-out text — its
token types deterministically encode 1–3-letter plaintext chunks, so it is a
wrong-hypothesis control (partial decode of real language) on which the
system must abstain, NOT a strict negative: its homophonic-head structure
margins are content-inflated by ≈ 0.27 bits/char relative to letter-shuffled
source twins (``voynichesque_nocontent``, 2026-08-31,
docs/project_status.md §5.6). Corrected 2026-09-01; originally read
"structured gibberish with no recoverable plaintext". Wraps the pinned
upstream script (same repo as naibbe_v2).
"""

from __future__ import annotations

import importlib.util
import os
import sys

from ..normalize import normalize
from .external import naibbe_repo


class Voynichesque:
    def __init__(self):
        repo = naibbe_repo()
        cwd = os.getcwd()
        try:
            os.chdir(repo)  # upstream reads references/*.csv relative to cwd
            spec = importlib.util.spec_from_file_location(
                "voynichesque", repo / "voynichesque.py"
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["voynichesque"] = mod
            spec.loader.exec_module(mod)
            self._options = mod.load_options(mod.CSV_PATH)
        finally:
            os.chdir(cwd)
        self._mod = mod

    def generate(self, text: str, seed: int = 0) -> str:
        """One Voynichesque encryption of ``text`` with sampled parameters.

        The input's content is destroyed by construction (random alphabets and
        re-tokenization); output is space-delimited glyph tokens — strip
        whitespace downstream like every other stream (design §2).
        """
        ciphertext, _alphabets, _params = self._mod.run_voynichesque_once(
            normalize(text), self._options, rng_seed=seed
        )
        return ciphertext
