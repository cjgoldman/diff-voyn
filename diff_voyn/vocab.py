"""Frozen vocabulary spec — task 0.1.

Resolution of the alphabet discrepancy (design §2, §10; cross-cutting X.4)
=========================================================================

The differentiable-inverse note called the Naibbe plaintext alphabet
"~22-letter" while listing 23 symbols. Resolved against Greshko's actual
tables (``references/naibbe_tables.csv`` in greshko/naibbe-cipher @ df3d074,
the repo behind Zenodo 10.5281/zenodo.16415087): all 18 tables
(unigram/prefix/suffix × alpha/beta1/beta2/beta3/gamma1/gamma2) cover exactly
**23 letters** each::

    a b c d e f g h i l m n o p q r s t u v x y z

i.e. the 26-letter Latin alphabet minus J, K, W — with U and V as *distinct*
cipher letters. The "~22" wording was an approximation; 23 is correct.

Frozen decisions (task 0.1)
---------------------------

- **K/W extension** (design §2): K and W are first-class vocabulary letters,
  needed by Germanic candidates. Greshko's own ``clean_line`` maps K→C and
  W→UU; that lossy mapping is a property of *the Naibbe cipher's* preprocessing
  and is applied only inside the Naibbe wrapper (``diff_voyn.ciphers``), never
  in the shared model pipeline — a lossy shared mapping would deflate Germanic
  likelihoods (R1 violation).
- **i/j**: J is merged into I in normalization (medieval convention; no Naibbe
  table has a j). J is not in the vocabulary.
- **u/v**: kept **distinct**, as in the source editions — the Naibbe tables
  encode u and v separately, so a u/v merge would orphan the v tables.
- **No SPACE token**: all whitespace is stripped in preprocessing (design §2);
  no space symbol exists in the vocabulary.
- **Specials**: PAD, MASK, NULL, BOS, EOS, plus LANG — the reserved
  conditioning slot from design §2 (unused by the additive-embedding
  conditioning of design §4, reserved so the vocab need not change if a
  prefix-token variant is ever ablated).

Total: 6 specials + 25 letters + 1 reserved = 32 (hardware-aligned).

This module is the single source of truth. ``spec_dict()``/``spec_hash()``
provide the versioned, hashable spec recorded in every run manifest.
"""

from __future__ import annotations

import hashlib
import json

VOCAB_VERSION = "v1"

# The 23 letters attested in Greshko's Naibbe tables (repo @ df3d074).
NAIBBE_LETTERS: str = "abcdefghilmnopqrstuvxyz"

# Frozen model alphabet: Naibbe 23 + K/W extension for Germanic (design §2).
LETTERS: str = "abcdefghiklmnopqrstuvwxyz"  # 26-letter Latin alphabet minus j

PAD = "<pad>"
MASK = "<mask>"
NULL = "<null>"
BOS = "<bos>"
EOS = "<eos>"
LANG = "<lang>"  # reserved conditioning slot (design §2/§4); unused in v1
RESERVED = "<res0>"

SPECIALS: list[str] = [PAD, MASK, NULL, BOS, EOS, LANG]

# id layout: specials first, then letters, then padding-to-32 reserved slots.
TOKENS: list[str] = SPECIALS + list(LETTERS) + [RESERVED]
assert len(TOKENS) == 32, f"vocab must be exactly 32 symbols, got {len(TOKENS)}"

TOKEN_TO_ID: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}
ID_TO_TOKEN: dict[int, str] = {i: t for t, i in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID[PAD]
MASK_ID = TOKEN_TO_ID[MASK]
NULL_ID = TOKEN_TO_ID[NULL]
BOS_ID = TOKEN_TO_ID[BOS]
EOS_ID = TOKEN_TO_ID[EOS]
LANG_ID = TOKEN_TO_ID[LANG]

LETTER_IDS: list[int] = [TOKEN_TO_ID[c] for c in LETTERS]
VOCAB_SIZE = 32


def encode(text: str) -> list[int]:
    """Map a normalized character stream to token ids.

    Raises ``KeyError`` on any character outside the frozen alphabet — inputs
    must already have passed :func:`diff_voyn.normalize.normalize`.
    """
    return [TOKEN_TO_ID[c] for c in text]


def decode(ids: list[int]) -> str:
    """Inverse of :func:`encode`; drops special tokens."""
    letters = set(LETTERS)
    out = []
    for i in ids:
        t = ID_TO_TOKEN[i]
        if t in letters:
            out.append(t)
    return "".join(out)


def spec_dict() -> dict:
    """The complete frozen spec as a JSON-serializable dict."""
    return {
        "vocab_version": VOCAB_VERSION,
        "vocab_size": VOCAB_SIZE,
        "tokens": TOKENS,
        "letters": LETTERS,
        "naibbe_letters": NAIBBE_LETTERS,
        "specials": SPECIALS,
        "notes": {
            "space": "no SPACE token; whitespace stripped in preprocessing (design §2)",
            "i_j": "j merged into i during normalization; j not in vocab",
            "u_v": "u and v distinct (both attested as separate Naibbe cipher letters)",
            "k_w": "first-class letters (Germanic extension); Naibbe-side k->c, w->uu "
            "mapping lives only in the cipher wrapper",
            "alphabet_provenance": "greshko/naibbe-cipher @ df3d074, "
            "references/naibbe_tables.csv (Zenodo 10.5281/zenodo.16415087): "
            "18 tables x 23 letters",
        },
    }


def spec_hash() -> str:
    """Stable sha256 of the frozen spec (recorded in run manifests, task 0.6)."""
    blob = json.dumps(spec_dict(), sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()
