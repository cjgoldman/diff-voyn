"""Build references/polygraphia_tables_v1.csv from the adjudicated transcripts
(docs/polygraphia_digitization_scope.md §3.2).

Source precedence per column: transcripts/merged/ (adjudicated) if present,
else pass1 provided pass1 == pass2 on every row (the validator must report
zero unresolved disagreements first — this script refuses otherwise).

Normalization applied here, recorded not silent:
- NFC, whitespace collapsed, lowercased for `word`;
- e-caudata (ę) -> e;
- nasal tilde `X~` expanded to Xn (Xm before b/m/p) — the printed form is
  kept in `raw` whenever it differs from `word` (case aside).

Output:
- references/polygraphia_tables_v1.csv          book,column,letter,word,raw
- references/polygraphia_annotations_v1.csv     book,column,letter,text,side,cut
- transcripts/build_report.json                 counts + flags
"""

import csv
import json
import os
import pathlib
import re
import sys
import unicodedata

DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/workspace/data"))
TR = DATA_ROOT / "external" / "polygraphia" / "transcripts"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "references"

KEYS = list("abcdefghiklmnopqrstvxyz") + ["w"]


def expand(text: str) -> tuple[str, list[str]]:
    """Lowercase, collapse spaces, ę->e, expand nasal tildes. Returns
    (expanded, unhandled_marks)."""
    t = unicodedata.normalize("NFC", text.strip()).lower()
    t = re.sub(r"\s+", " ", t).replace("ę", "e")
    def _sub(m: re.Match) -> str:
        letter = m.group(1)
        follower = m.group(2)
        nasal = "m" if follower in "bmp" else "n"
        return letter + nasal + follower

    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"([aeiouy])~(.)", _sub, t)
    t = re.sub(r"([aeiouy])~$", r"\1n", t)  # word-final tilde -> n
    unhandled = re.findall(r"\S*~\S*", t)
    return t, unhandled


def main() -> None:
    p1 = {p.stem: p for p in (TR / "pass1").glob("*.json")}
    p2 = {p.stem: p for p in (TR / "pass2").glob("*.json")}
    merged = (
        {p.stem: p for p in (TR / "merged").glob("*.json")}
        if (TR / "merged").exists()
        else {}
    )

    cols = sorted(set(p1) & set(p2) | set(merged))
    report = {
        "columns": len(cols),
        "rows": 0,
        "raw_differs": 0,
        "unhandled_marks": [],
        "flags": [],
    }
    table_rows, ann_rows = [], []

    for col in cols:
        if col in merged:
            data = json.loads(merged[col].read_text())
        else:
            d1 = json.loads(p1[col].read_text())
            d2 = json.loads(p2[col].read_text())
            t1 = [r.get("text", "") for r in d1["rows"]]
            t2 = [r.get("text", "") for r in d2["rows"]]
            n1 = [expand(t)[0] for t in t1]
            n2 = [expand(t)[0] for t in t2]
            if n1 != n2:
                print(
                    f"REFUSING: {col} has unresolved pass disagreements; "
                    f"run polygraphia_validate.py and adjudicate first."
                )
                sys.exit(1)
            data = d1
        book = col.split("_")[0]
        colnum = int(col.split("col")[1])
        rows = data["rows"]
        if [r["key"] for r in rows] != KEYS:
            report["flags"].append(f"{col}: non-canonical keys")
        for r in rows:
            word, unh = expand(r["text"])
            report["unhandled_marks"] += [f"{col}:{r['key']}:{u}" for u in unh]
            raw = r["text"].strip()
            raw_out = raw if raw.lower() != word else ""
            if raw_out:
                report["raw_differs"] += 1
            table_rows.append([book, colnum, r["key"], word, raw_out])
            report["rows"] += 1
        words = [expand(r["text"])[0] for r in rows]
        dupes = {w for w in words if words.count(w) > 1}
        if dupes:
            report["flags"].append(
                f"{col}: duplicate words within column {sorted(dupes)}"
            )
        for a in data.get("annotations", []):
            ann_rows.append(
                [
                    book,
                    colnum,
                    a.get("key"),
                    a.get("text", ""),
                    a.get("side", ""),
                    a.get("cut", False),
                ]
            )

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "polygraphia_tables_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["book", "column", "letter", "word", "raw"])
        w.writerows(table_rows)
    with open(OUT_DIR / "polygraphia_annotations_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["book", "column", "letter", "text", "side", "cut"])
        w.writerows(ann_rows)
    (TR / "build_report.json").write_text(json.dumps(report, indent=1))
    print(
        f"{report['rows']} rows over {report['columns']} columns "
        f"({report['raw_differs']} raw!=word, "
        f"{len(report['unhandled_marks'])} unhandled marks, "
        f"{len(ann_rows)} annotations, {len(report['flags'])} flags)"
    )
    for fl in report["flags"][:10]:
        print("  flag:", fl)


if __name__ == "__main__":
    main()
