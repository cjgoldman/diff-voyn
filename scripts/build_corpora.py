"""Assemble the normalized corpus and carve held-out splits (tasks 0.2–0.4).

Run: ``uv run python scripts/build_corpora.py``

Reads raw sources (voynich-attack CSVs for Latin/German, downloaded texts for
Italian), applies the shared normalizer, writes per-document normalized
streams + the corpus manifest (the task-0.2 table), then carves the versioned
held-out splits (task 0.4). Prints the acceptance numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.assemble import (
    CORPUS_VERSION,
    assemble_language,
    discover_documents,
    write_manifest,
)
from diff_voyn.corpus.splits import carve_splits


def main() -> None:
    root = data_root()
    docs = discover_documents(root)
    records, stats = {}, {}
    for lang, lang_docs in docs.items():
        print(f"assembling {lang}: {len(lang_docs)} documents ...", flush=True)
        records[lang], stats[lang] = assemble_language(root, lang, lang_docs)
        total = sum(r.norm_chars for r in records[lang])
        s = stats[lang]
        print(
            f"  {lang}: {len(records[lang])} docs kept, {total:,} normalized chars, "
            f"letter-drop rate {s.letter_drop_rate:.5%} "
            f"({'OK' if s.letter_drop_rate < 0.001 else 'FAIL >0.1%'})"
        )
    manifest_path = write_manifest(root, records, stats)
    print(f"manifest: {manifest_path}")

    splits_path = carve_splits(manifest_path)
    import json

    splits = json.loads(splits_path.read_text())
    for lang, sp in splits["languages"].items():
        ok = sp["heldout_chars"] >= 200_000
        print(
            f"  split {lang}: heldout {sp['heldout_chars']:,} chars "
            f"({len(sp['heldout'])} docs, {'OK' if ok else 'FAIL <200k'}), "
            f"train {sp['train_chars']:,} chars ({len(sp['train'])} docs)"
        )
    print(f"splits: {splits_path}")
    print(f"corpus version: {CORPUS_VERSION}")


if __name__ == "__main__":
    main()
