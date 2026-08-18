"""VMS ingest — task 0.8.

Chosen transcriptions (both IVTFF, downloaded from voynich.nu/data by
``scripts/fetch_external.py``):

- ``IT2a-n.txt`` — Takahashi (EvaT), the completeness reference.
- ``RF1b-e.txt`` — the Reference transliteration, named in the task.

(The independent voynich-attack transcription remains available via
``voynpy.corpora`` for cross-checks; its custom glyph alphabet keeps it out of
the EVA pipeline.)

Parsing policy (frozen here; uncertainty accounting is retained in counts):

- Page headers ``<fNN>  <! ... $L=A ... >`` supply the Currier language; pages
  without ``$L`` are kept in a separate "unassigned" stream — **A and B are
  never pooled** (design §9).
- Inline comments/markup ``<...>`` are removed; alternate readings ``[a:o]``
  take the first alternative; ``?`` (uncertain glyph) and extended-EVA
  ``@nnn;`` codes are dropped and counted (uncertain-glyph policy: exclude,
  never guess).
- Word separators ``.``/``,`` are treated as whitespace: word and character
  counts are taken *before* stripping (to reconcile with published figures),
  then all separators are removed (task 0.3 / design §2) for the model-facing
  streams.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_PAGE_RE = re.compile(r"^<(f[^.>]+|[^.>]+)>\s*(<!.*>)?\s*$")
_LOCUS_RE = re.compile(r"^<([^,>]+)\.([^,>]+),([^>]*)>\s*(.*)$")
_CURRIER_RE = re.compile(r"\$L=([A-Z])")
_INLINE_MARKUP_RE = re.compile(r"<[^>]*>")
_ALT_READING_RE = re.compile(r"\[([^:\]]*):[^\]]*\]")
_EXTENDED_EVA_RE = re.compile(r"@\d+;")


def parse_ivtff(path: Path) -> list[dict]:
    """Parse an IVTFF file into [{page, currier, lines: [locus text, ...]}]."""
    pages: list[dict] = []
    current: dict | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        m = _PAGE_RE.match(raw)
        if m and not _LOCUS_RE.match(raw):
            cm = _CURRIER_RE.search(m.group(2) or "")
            current = {
                "page": m.group(1),
                "currier": cm.group(1) if cm else None,
                "lines": [],
            }
            pages.append(current)
            continue
        lm = _LOCUS_RE.match(raw)
        if lm and current is not None:
            current["lines"].append(lm.group(4))
    return pages


def _clean_line(text: str, dropped: Counter) -> str:
    text = _ALT_READING_RE.sub(r"\1", text)
    text = _INLINE_MARKUP_RE.sub("", text)
    n_ext = len(_EXTENDED_EVA_RE.findall(text))
    if n_ext:
        dropped["extended_eva"] += n_ext
        text = _EXTENDED_EVA_RE.sub("", text)
    n_unc = text.count("?")
    if n_unc:
        dropped["uncertain_glyph"] += n_unc
        text = text.replace("?", "")
    return text


def build_dialect_streams(path: Path) -> dict:
    """Parse, split by Currier dialect, count, and strip separators.

    Returns ``{"streams": {A, B, unassigned}, "counts": {...}}`` where counts
    are taken before separator stripping (for reconciliation with published
    figures) and after (the model-facing character counts).
    """
    pages = parse_ivtff(path)
    dropped: Counter = Counter()
    streams: dict[str, list[str]] = {"A": [], "B": [], "unassigned": []}
    pre_counts = {k: {"words": 0, "chars": 0, "pages": 0} for k in streams}

    for page in pages:
        key = page["currier"] if page["currier"] in ("A", "B") else "unassigned"
        page_words = 0
        page_chars = 0
        for line in page["lines"]:
            cleaned = _clean_line(line, dropped)
            words = [w for w in re.split(r"[.,\s]+", cleaned) if w]
            page_words += len(words)
            page_chars += sum(len(w) for w in words)
            streams[key].append("".join(words))
        pre_counts[key]["words"] += page_words
        pre_counts[key]["chars"] += page_chars
        pre_counts[key]["pages"] += 1

    joined = {k: "".join(v) for k, v in streams.items()}
    return {
        "streams": joined,
        "counts": {
            "pre_strip": pre_counts,
            "post_strip_chars": {k: len(v) for k, v in joined.items()},
            "dropped": dict(dropped),
            "total_words": sum(c["words"] for c in pre_counts.values()),
            "total_chars_pre_strip": sum(c["chars"] for c in pre_counts.values()),
        },
    }


def ingest_to(data_root: Path, source: Path, name: str) -> Path:
    """Run ingest and persist streams + counts under DATA_ROOT/vms/<name>/."""
    out_dir = data_root / "vms" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_dialect_streams(source)
    for dialect, stream in result["streams"].items():
        (out_dir / f"currier_{dialect}.txt").write_text(stream, encoding="utf-8")
    (out_dir / "counts.json").write_text(json.dumps(result["counts"], indent=2))
    return out_dir
