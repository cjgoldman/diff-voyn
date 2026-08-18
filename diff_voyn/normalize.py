"""Shared normalization pipeline — task 0.3.

One pipeline, applied **identically** to every text stream (training corpora,
synthetic ciphertexts, partial decipherments, VMS transliterations). No
per-language rules anywhere (R1: a mapping that survives better for one
language hands it an unearned likelihood edge).

Stages, per character of NFC-normalized, casefolded input:

1. **Whitespace** (anything ``str.isspace`` plus unicode separators) is
   removed. Counted separately — whitespace removal is a modeling decision
   (design §2) and excluded from the drop budget.
2. **Compatibility folding**: casefold + NFKD expands ligatures and sharp s
   (ß→ss, æ→ae, œ→oe, ﬁ→fi, …) and splits diacritics into combining marks,
   which are stripped (ü→u, é→e, ñ→n, …). Applied uniformly to all languages —
   the German-convention ü→ue expansion would be a per-language mapping and is
   deliberately *not* used.
3. **Letter remaps** (uniform, deterministic): j→i (medieval convention;
   design §2), long s ſ→s.
4. **Punctuation, digits, and symbols** are removed and counted as
   ``punct_digit_removed`` — an expected, documented category (source editions
   carry editorial punctuation the ciphers never see).
5. **Foreign-script letters** (Greek, Hebrew, Cyrillic, … — mostly embedded
   quotations in Latin scientific works) are removed and counted as
   ``foreign_script_removed``. This is a *by-design* category, applied
   identically to every language: the frozen 25-letter alphabet cannot
   represent them, and transliteration would be a per-language lossy mapping
   (an R1 violation — the design treats e.g. "Greek (transliterated)" as its
   own candidate language, not as part of Latin). Rates are reported per
   language in the corpus manifest, not hidden.
6. Latin-script letters that still aren't one of the 25 frozen letters after
   all folds are dropped and counted as ``letters_dropped`` — true pipeline
   lossiness, the number under the acceptance budget.

Acceptance accounting (task 0.3): ``letters_dropped`` must stay below 0.1% of
alphabetic input characters per language; zero whitespace survives; the
pipeline is idempotent (``normalize(normalize(x)) == normalize(x)``).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import cache

from .vocab import LETTERS

NORMALIZER_VERSION = "v1"

_LETTER_SET = frozenset(LETTERS)
_REMAP = {"j": "i", "ſ": "s"}
# æ/œ have no Unicode decomposition, so NFKD leaves them intact; expand
# explicitly (uniformly for all languages) before per-char classification.
# Medieval/early-print abbreviature and variant letters (attested in the DTA
# and Corpus Corporum sources) get their standard letter expansions — these
# are Latin-script letters the drop budget must not lose:
_EXPAND = {
    "æ": "ae",
    "œ": "oe",
    "ꝛ": "r",  # r rotunda (ubiquitous in early German prints)
    "ꝙ": "q",  # q with diagonal stroke (quod abbreviature)
    "ꝗ": "q",  # q with stroke through descender
    "ʒ": "z",  # ezh, used for z / tailed z in FNHD prints
    "ǯ": "z",
    "ꝟ": "v",  # v with diagonal stroke
    "đ": "d",  # d with stroke
    "ð": "d",
    "ↄ": "c",  # reversed c (con- abbreviature)
    "ø": "o",
    "ı": "i",  # dotless i
    "ᵱ": "p",
    "ꝓ": "p",
    "ꝑ": "p",
    "ȝ": "g",  # yogh
    "ɔ": "o",
    "ɯ": "m",
    "ȣ": "ou",  # ou ligature
}


@dataclass
class NormStats:
    """Per-category character accounting for one normalization pass."""

    input_chars: int = 0
    kept: int = 0
    whitespace_removed: int = 0
    punct_digit_removed: int = 0
    letters_dropped: int = 0
    foreign_script_removed: int = 0
    dropped_examples: dict[str, int] = field(default_factory=dict)
    foreign_examples: dict[str, int] = field(default_factory=dict)

    @property
    def alphabetic_input(self) -> int:
        return self.kept + self.letters_dropped

    @property
    def letter_drop_rate(self) -> float:
        """Latin-script letters lost as a fraction of alphabetic input — the
        0.1% acceptance number (foreign-script removal is a separate,
        by-design category; see module docstring)."""
        n = self.alphabetic_input
        return self.letters_dropped / n if n else 0.0

    @property
    def foreign_script_rate(self) -> float:
        n = self.kept + self.letters_dropped + self.foreign_script_removed
        return self.foreign_script_removed / n if n else 0.0

    def merge(self, other: NormStats) -> None:
        self.input_chars += other.input_chars
        self.kept += other.kept
        self.whitespace_removed += other.whitespace_removed
        self.punct_digit_removed += other.punct_digit_removed
        self.letters_dropped += other.letters_dropped
        self.foreign_script_removed += other.foreign_script_removed
        for ch, n in other.dropped_examples.items():
            self.dropped_examples[ch] = self.dropped_examples.get(ch, 0) + n
        for ch, n in other.foreign_examples.items():
            self.foreign_examples[ch] = self.foreign_examples.get(ch, 0) + n

    def as_dict(self) -> dict:
        def top(d: dict[str, int]) -> dict[str, int]:
            return dict(sorted(d.items(), key=lambda kv: -kv[1])[:20])

        return {
            "input_chars": self.input_chars,
            "kept": self.kept,
            "whitespace_removed": self.whitespace_removed,
            "punct_digit_removed": self.punct_digit_removed,
            "letters_dropped": self.letters_dropped,
            "letter_drop_rate": self.letter_drop_rate,
            "foreign_script_removed": self.foreign_script_removed,
            "foreign_script_rate": self.foreign_script_rate,
            "top_dropped": top(self.dropped_examples),
            "top_foreign": top(self.foreign_examples),
        }


# Outcome tags for the per-character memo table.
_KEEP, _WS, _PUNCT, _DROP, _FOREIGN = 0, 1, 2, 3, 4


def _is_latin_script(c: str) -> bool:
    return "LATIN" in unicodedata.name(c, "")


@cache
def _fold_char(ch: str) -> tuple[int, str]:
    """Classify one input character and return (outcome, replacement)."""
    if ch.isspace() or unicodedata.category(ch) == "Zs":
        return _WS, ""
    folded = ch.casefold()  # ß→ss, İ→i̇, uppercase→lowercase
    folded = "".join(_EXPAND.get(c, c) for c in folded)
    decomposed = unicodedata.normalize("NFKD", folded)
    out = []
    saw_latin_letter = False
    saw_foreign_letter = False
    for c in decomposed:
        cat = unicodedata.category(c)
        if cat.startswith("M"):  # combining marks: strip diacritics
            continue
        if c.isspace() or cat == "Zs":
            continue
        c = _REMAP.get(c, c)
        if c in _LETTER_SET:
            out.append(c)
            saw_latin_letter = True
        elif cat.startswith("L"):
            if _is_latin_script(c):
                saw_latin_letter = True  # unmapped Latin letter: real loss
            else:
                saw_foreign_letter = True  # Greek/Hebrew/...: by-design removal
    if out:
        return _KEEP, "".join(out)
    if saw_latin_letter:
        return _DROP, ""
    if saw_foreign_letter:
        return _FOREIGN, ""
    return _PUNCT, ""


def normalize(text: str, stats: NormStats | None = None) -> str:
    """Normalize ``text`` to the frozen alphabet, updating ``stats`` if given."""
    text = unicodedata.normalize("NFC", text)
    if stats is None:
        stats = NormStats()
    stats.input_chars += len(text)
    out = []
    for ch in text:
        outcome, rep = _fold_char(ch)
        if outcome == _KEEP:
            out.append(rep)
            stats.kept += len(rep)
        elif outcome == _WS:
            stats.whitespace_removed += 1
        elif outcome == _PUNCT:
            stats.punct_digit_removed += 1
        elif outcome == _FOREIGN:
            stats.foreign_script_removed += 1
            stats.foreign_examples[ch] = stats.foreign_examples.get(ch, 0) + 1
        else:
            stats.letters_dropped += 1
            stats.dropped_examples[ch] = stats.dropped_examples.get(ch, 0) + 1
    return "".join(out)
