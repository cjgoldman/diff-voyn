"""Structural validation + double-key diff for the Polygraphia transcription
(docs/polygraphia_digitization_scope.md §4.3).

Checks every transcript JSON for shape (exactly 24 rows, canonical key
order) and diffs pass1 against pass2 row by row. Disagreements go to
transcripts/disagreements.json for adjudication; adjudicated rows are
written to transcripts/merged/<column>.json by the adjudicator (same schema
as a pass file, plus "adjudicated": true).

Usage:
  uv run python scripts/polygraphia_validate.py            # report
  uv run python scripts/polygraphia_validate.py --columns b1_col001,b1_col002
"""

import argparse
import json
import os
import pathlib
import re
import unicodedata

DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/workspace/data"))
TR = DATA_ROOT / "external" / "polygraphia" / "transcripts"

KEYS = list("abcdefghiklmnopqrstvxyz") + ["w"]


def canon(text: str) -> str:
    """Comparison form: case-folded, single-spaced, combining marks joined."""
    t = unicodedata.normalize("NFC", text.strip().lower())
    t = re.sub(r"\s+", " ", t)
    t = t.replace("ę", "e")
    return t


def check_shape(col: str, data: dict) -> list[str]:
    errs = []
    rows = data.get("rows", [])
    if len(rows) != 24:
        errs.append(f"{col}: {len(rows)} rows (want 24)")
    keys = [r.get("key") for r in rows]
    if keys != KEYS[: len(keys)]:
        bad = [f"{i}:{k}" for i, (k, want) in enumerate(zip(keys, KEYS)) if k != want]
        errs.append(f"{col}: key order broken at {bad[:5]}")
    for r in rows:
        if not r.get("text", "").strip() and r.get("key") not in data.get(
            "illegible", []
        ):
            errs.append(f"{col}: empty text at key {r.get('key')}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", default=None)
    args = ap.parse_args()

    p1 = {p.stem: p for p in (TR / "pass1").glob("*.json")}
    p2 = {p.stem: p for p in (TR / "pass2").glob("*.json")}
    merged = (
        {p.stem for p in (TR / "merged").glob("*.json")}
        if (TR / "merged").exists()
        else set()
    )

    cols = sorted(set(p1) | set(p2))
    if args.columns:
        cols = [c for c in cols if c in set(args.columns.split(","))]

    shape_errs, disagreements = [], {}
    n_rows = n_diff = 0
    for col in cols:
        d1 = json.loads(p1[col].read_text()) if col in p1 else None
        d2 = json.loads(p2[col].read_text()) if col in p2 else None
        for d in (d1, d2):
            if d:
                shape_errs += check_shape(col, d)
        if not (d1 and d2):
            continue
        diffs = []
        for r1, r2 in zip(d1["rows"], d2["rows"]):
            n_rows += 1
            if canon(r1.get("text", "")) != canon(r2.get("text", "")):
                diffs.append(
                    {
                        "key": r1.get("key"),
                        "pass1": r1.get("text"),
                        "pass2": r2.get("text"),
                    }
                )
        if diffs:
            n_diff += len(diffs)
            disagreements[col] = diffs

    out = TR / "disagreements.json"
    out.write_text(
        json.dumps(disagreements, indent=1, ensure_ascii=False, sort_keys=True)
    )

    print(
        f"columns: {len(cols)} (pass1 {len(p1)}, pass2 {len(p2)}, merged {len(merged)})"
    )
    print(f"shape errors: {len(shape_errs)}")
    for e in shape_errs[:20]:
        print("  ", e)
    both = len(set(p1) & set(p2))
    print(
        f"double-keyed: {both} columns, {n_rows} rows, {n_diff} disagreements "
        f"({100*n_diff/max(n_rows,1):.2f}%) -> {out}"
    )
    unresolved = [c for c in disagreements if c not in merged]
    print(f"columns needing adjudication: {len(unresolved)}")


if __name__ == "__main__":
    main()
