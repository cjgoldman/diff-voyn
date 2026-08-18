"""Corpus assembly — tasks 0.2 (inventory) and 0.3 (normalization application).

Frozen language inventory (task 0.2): **Latin, Italian, German.**

Sources:

- Latin and German come from the reference corpora bundled in Boxer's
  voynich-attack repo (cloned under ``DATA_ROOT/external/voynich-attack``,
  pinned by ``scripts/fetch_external.py``): sentence-level CSVs with a
  ``textstring_orig`` column and a ``build.py`` documenting upstream source.
  We consume ``textstring_orig`` (verbatim source text) and apply *our* shared
  normalizer — never the repo's pre-normalized ``textstring_simple``, whose
  u→v folding is exactly the kind of per-language lossy mapping R1 forbids.
- Italian is assembled from public-domain period texts downloaded to
  ``DATA_ROOT/raw/italian`` (see that directory's ``manifest.csv``).

A *document* is one work (one CSV file / one text file). Output layout::

    DATA_ROOT/corpora/<version>/<language>/docs/<doc_id>.txt   # normalized
    DATA_ROOT/corpora/<version>/manifest.json                  # corpus table

Exact-duplicate documents (identical normalized sha256) are dropped, keeping
the first occurrence — this also guarantees no duplicate can straddle the
train/held-out boundary (task 0.3 acceptance).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..normalize import NORMALIZER_VERSION, NormStats, normalize
from ..vocab import VOCAB_VERSION

csv.field_size_limit(sys.maxsize)

CORPUS_VERSION = "v1"
LANGUAGES = ["latin", "italian", "german"]

# Upstream licensing by source group (voynich-attack README: "each reference
# text retains its upstream license"); italian licenses come from its manifest.
SOURCE_LICENSES = {
    "DTA": "CC BY-SA 4.0 (Deutsches Textarchiv)",
    "zeno": "public domain content via zeno.org",
    "legacy": "per-text, see build.py in source dir (voynich-attack)",
    "CorpusCorporum": "per-text, mlat.uzh.ch (research use)",
    "apothecary": "public domain (pre-1900 edition)",
}


@dataclass
class DocRecord:
    doc_id: str
    language: str
    domain: str  # source group: DTA / zeno / legacy / CorpusCorporum / gutenberg...
    source_path: str
    license: str
    raw_chars: int
    norm_chars: int
    sha256: str


def _slug(*parts: str) -> str:
    s = "_".join(parts)
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()


def _read_va_csv(path: Path) -> str:
    """Concatenate the verbatim-text column of a voynich-attack corpus CSV.

    Prefers ``textstring_orig`` (untouched source text); legacy files carry
    only ``textstring``; a few legacy files have unnamed columns, where the
    longest field per row is the text.
    """
    chunks: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        col = None
        for name in ("textstring_orig", "textstring"):
            if name in header:
                col = header.index(name)
                break
        for row in reader:
            if not row:
                continue
            t = row[col] if col is not None and col < len(row) else max(row, key=len)
            if t:
                chunks.append(t)
    return "\n".join(chunks)


# Legacy dirs bundle several processing variants of the same work; pick one.
# _lat0 variants apply a German-specific ue-expansion we must not inherit
# (R1); *_edited is the curated pass; base files keep source orthography.
def _legacy_variant_rank(path: Path) -> tuple[int, str]:
    n = path.name.lower()
    if "firstpass" in n:
        rank = 5
    elif n.endswith("_edited.csv"):
        rank = 0
    elif "_lat1" in n:
        rank = 2
    elif "_lat0" in n:
        rank = 3
    elif "_v0" in n:
        rank = 4
    else:
        rank = 1
    return rank, path.name


def _select_documents(base: Path, domain: str, paths: list[Path]) -> list[Path]:
    if domain != "legacy":
        return paths
    by_work: dict[Path, list[Path]] = {}
    for p in paths:
        by_work.setdefault(p.parent, []).append(p)
    return sorted(min(files, key=_legacy_variant_rank) for files in by_work.values())


def discover_documents(data_root: Path) -> dict[str, list[dict]]:
    """Enumerate raw documents per language: {lang: [{doc_id, path, domain, license, reader}]}."""
    va = data_root / "external" / "voynich-attack" / "corpora"
    out: dict[str, list[dict]] = {lang: [] for lang in LANGUAGES}

    for lang in ("latin", "german"):
        base = va / lang
        if not base.is_dir():
            raise FileNotFoundError(
                f"{base} missing — run scripts/fetch_external.py first"
            )
        by_domain: dict[str, list[Path]] = {}
        for path in sorted(base.rglob("*.csv")):
            by_domain.setdefault(path.relative_to(base).parts[0], []).append(path)
        for domain, paths in sorted(by_domain.items()):
            for path in _select_documents(base, domain, paths):
                out[lang].append(
                    {
                        "doc_id": _slug(domain, *path.relative_to(base / domain).parts)[
                            :120
                        ],
                        "path": path,
                        "domain": domain,
                        "license": SOURCE_LICENSES.get(domain, "unknown"),
                        "format": "va_csv",
                    }
                )

    it_dir = data_root / "raw" / "italian"
    it_manifest = it_dir / "manifest.csv"
    licenses: dict[str, str] = {}
    if it_manifest.exists():
        with open(it_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                licenses[row["filename"]] = row.get("license", "public domain")
    for path in sorted(it_dir.glob("*.txt")):
        out["italian"].append(
            {
                "doc_id": _slug(path.stem)[:120],
                "path": path,
                "domain": "gutenberg_wikisource",
                "license": licenses.get(path.name, "public domain"),
                "format": "txt",
            }
        )
    return out


def assemble_language(
    data_root: Path, language: str, docs: Iterable[dict]
) -> tuple[list[DocRecord], NormStats]:
    """Normalize every document of one language and write the corpus files."""
    out_dir = data_root / "corpora" / CORPUS_VERSION / language / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[DocRecord] = []
    lang_stats = NormStats()
    seen_hashes: dict[str, str] = {}
    for doc in docs:
        raw = (
            _read_va_csv(doc["path"])
            if doc["format"] == "va_csv"
            else Path(doc["path"]).read_text(encoding="utf-8")
        )
        doc_stats = NormStats()
        norm = normalize(raw, doc_stats)
        if not norm:
            continue
        digest = hashlib.sha256(norm.encode()).hexdigest()
        if digest in seen_hashes:  # exact duplicate work: keep first
            continue
        seen_hashes[digest] = doc["doc_id"]
        lang_stats.merge(doc_stats)
        (out_dir / f"{doc['doc_id']}.txt").write_text(norm, encoding="utf-8")
        records.append(
            DocRecord(
                doc_id=doc["doc_id"],
                language=language,
                domain=doc["domain"],
                source_path=str(doc["path"]),
                license=doc["license"],
                raw_chars=len(raw),
                norm_chars=len(norm),
                sha256=digest,
            )
        )
    return records, lang_stats


def write_manifest(
    data_root: Path,
    records: dict[str, list[DocRecord]],
    stats: dict[str, NormStats],
    extra: dict | None = None,
) -> Path:
    """Write the corpus manifest: the task-0.2 table plus per-language norm stats."""
    low_resource_threshold = 2_000_000
    per_language = {}
    for lang, recs in records.items():
        total = sum(r.norm_chars for r in recs)
        per_language[lang] = {
            "documents": len(recs),
            "norm_chars": total,
            "low_resource_flag": total < low_resource_threshold,
            "domains": sorted({r.domain for r in recs}),
            "licenses": sorted({r.license for r in recs}),
            "norm_stats": stats[lang].as_dict(),
        }
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "vocab_version": VOCAB_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "languages": per_language,
        "documents": {lang: [asdict(r) for r in records[lang]] for lang in records},
    }
    if extra:
        manifest.update(extra)
    path = data_root / "corpora" / CORPUS_VERSION / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
