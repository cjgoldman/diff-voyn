"""VMS ingest run (task 0.8): parse both chosen transliterations, split by
Currier dialect, record pre/post-strip counts.

Run: ``uv run python scripts/ingest_vms.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import data_root
from diff_voyn.vms.ingest import ingest_to

SOURCES = {"takahashi_IT2a": "IT2a-n.txt", "reference_RF1b": "RF1b-e.txt"}


def main() -> None:
    root = data_root()
    for name, filename in SOURCES.items():
        src = root / "raw" / "vms" / filename
        out = ingest_to(root, src, name)
        counts = json.loads((out / "counts.json").read_text())
        pre = counts["pre_strip"]
        print(
            f"{name}: total {counts['total_words']:,} words / "
            f"{counts['total_chars_pre_strip']:,} chars (pre-strip); "
            f"A: {pre['A']['pages']}p {pre['A']['words']:,}w, "
            f"B: {pre['B']['pages']}p {pre['B']['words']:,}w, "
            f"unassigned: {pre['unassigned']['pages']}p "
            f"{pre['unassigned']['words']:,}w; dropped {counts['dropped']}"
        )
        print(f"  post-strip chars: {counts['post_strip_chars']}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
