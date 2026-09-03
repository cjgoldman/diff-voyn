#!/usr/bin/env python3
"""Documentation coherence check — the mechanical half of the reading rule.

Guards against the drift the 2026-09-01 audit found: dated records being read
as current status. Deterministic, stdlib-only, < 1 s. Wired into
``.claude/settings.json`` (Stop + SessionStart hooks) and ``.githooks/pre-commit``.

Checks (each an ERROR unless noted):

  banner       every ``docs/*.md`` except ``project_status.md`` carries exactly one
               ``Record status (banner`` paragraph.
  status-doc   ``docs/project_status.md`` exists and has a ``Status date: YYYY-MM-DD`` line.
  status-fresh docs/CLAUDE.md/memory changed in the working tree (vs HEAD) or committed
               after project_status.md's status date ⇒ project_status.md must also be
               touched (its status date bumped, §1/§2/§5 updated — or the date bumped with
               a "(reviewed, no change)" note if the change is cosmetic). This is what
               makes the status date mean "last reviewed".
  phrases      superseded claims may appear only with their qualifier nearby (±4 lines):
               the "≥ 8 tokens/type" wall needs "plain SA"/stage wording, "judge in the
               acceptance rule … open path/not done" needs "no gain"/"tested", etc.
               The table lives in PHRASE_RULES below — extend it when a study supersedes
               a claim (and record the canonical reading in project_status.md §5).
  refs         every backticked ``docs/<name>.md`` reference in docs, CLAUDE.md, memory
               and scripts points at an existing file.
  sections     no duplicated ``## N.`` section numbers within a doc.
  memory       every memory file is indexed exactly once in MEMORY.md and every indexed
               file exists; memory ``description:`` lines obey the phrase rules.
  claude-md    CLAUDE.md has a "Current status (YYYY-MM-DD)" heading whose date is not
               older than project_status.md's status date.

Exit 0 when clean, 1 with a findings list otherwise. ``--hook stop`` / ``--hook
session-start`` emit the Claude Code hook JSON instead (see main()).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STATUS_DOC = DOCS / "project_status.md"
CLAUDE_MD = ROOT / "CLAUDE.md"
MEMORY_DIR = Path(
    os.environ.get(
        "CLAUDE_MEMORY_DIR",
        Path.home() / ".claude" / "projects" / "-workspace" / "memory",
    )
)

BANNER_MARK = "Record status (banner"
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# (name, trigger regex, qualifier regex that must appear within ±4 lines, message)
# Triggers are case-insensitive. Keep triggers narrow: they fire on prose, not code.
PHRASE_RULES = [
    (
        "wall-8",
        (
            r"(findab|recover|identifiab|wall|perfect|solv|unsolvable)[^.\n]{0,60}\b(≥\s?8|~8|>= ?8|8)\s?tok(ens?)?(/| per )type"
            r"|(≥\s?8|~8|>= ?8)\s?tok(ens?)?(/| per )type[^.\n]{0,60}(findab|recover|identifiab|wall|perfect|solv)"
        ),
        (
            r"plain[- ]SA|plain multi-restart|solver stage|by stage|current solver|solver of record|"
            r"≈ ?4|re-seeding|wildcard|superseded|originally|corrected|this study|baseline|stage 1|historical"
        ),
        (
            "the ≥ 8 tokens/type wall is the plain-SA figure — name the solver stage "
            "(current solver ≈ 4; docs/project_status.md §3)"
        ),
    ),
    (
        "judge-in-acceptance-open",
        (
            r"judge (in|inside) the (loop'?s )?acceptance rule[^.\n]{0,80}"
            r"(open path|only stronger path|only change|not done|remains|next step|must enter)"
        ),
        r"no gain|tested 2026-08-28|superseded|\[.*2026-08-28",
        (
            "judge-in-acceptance was tested 2026-08-28 with no gain — annotate "
            "(docs/project_status.md §5.2)"
        ),
    ),
    (
        "no-vms-cell",
        r"no VMS cell was run",
        r"superseded|72|24/24|8/8|2026-08-26",
        "VMS cells were run 08-26/08-29/08-31 — annotate (docs/project_status.md §5.3)",
    ),
    (
        "winners-curse",
        (
            r"(=|because|mechanism is|through|due to)\s*[^.\n]{0,50}winner'?s curse"
            r"|winner'?s curse[^.\n]{0,160}(fix before reuse|must be fixed|not fit for reuse|per-move)"
        ),
        r"choice[- ]term|choice-bits|objective|re-diagnos|superseded|corrected|not winner|race.polish",
        (
            "Borg polish failure is the choice term in the objective, not winner's curse "
            "(docs/project_status.md §5.4)"
        ),
    ),
    (
        "order0-charge",
        r"uncovered[^.\n]{0,40}order-0 entropy|order-0 entropy[^.\n]{0,40}uncovered",
        r"fallback|best held-out|n-gram cross-entropy|corrected|originally",
        (
            "uncovered symbols are charged at the best held-out n-gram cross-entropy "
            "(order-0 only as fallback; docs/project_status.md §5.5)"
        ),
    ),
    (
        "dirty5-replicate",
        r"dirty[- ]5 ?%[^.\n]{0,80}(did not replicate|not replicated)|(did not replicate|not replicated)[^.\n]{0,60}dirty",
        r"B[- ]shape|Blike|A[- ]shape|2026-09-03|09-03|§10\.6|superseded|5\.18",
        "the dirty-5 % German non-replication is the A-shape result; at B shape it is called 2/2 (docs/project_status.md §5.18)",
    ),
    (
        "recovered-at-3",
        r"recovers? (a )?key at 3(\.0)? tokens? per type",
        r"corrected|originally|nothing at|false|4\.1",
        "nothing at ≤ 3.5 tokens/type has ever been recovered (docs/project_status.md §5.9)",
    ),
]

REF_RE = re.compile(r"`docs/([A-Za-z0-9_\-]+\.md)`")
REF_ABSENT_OK = re.compile(
    r"does not exist|never (been )?written|not (yet )?written|to be written|never created|"
    r"not created|would be|planned|proposed|absent|missing|removed|deleted|superseded by|"
    r"\bwrite\b|write-up|go to|to-do|todo",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"^## (\d+[a-z]?)\.\s", re.MULTILINE)


class Findings:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return ""


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def check_banners(f: Findings, docs: list[Path]) -> None:
    for p in docs:
        if p == STATUS_DOC:
            continue
        n = p.read_text(encoding="utf-8").count(BANNER_MARK)
        if n != 1:
            f.err(
                f"banner: {rel(p)} has {n} '{BANNER_MARK}' paragraphs (need exactly 1)"
            )


def status_date(f: Findings) -> str | None:
    if not STATUS_DOC.exists():
        f.err("status-doc: docs/project_status.md is missing")
        return None
    head = STATUS_DOC.read_text(encoding="utf-8")[:2000]
    m = re.search(r"Status date:\s*\*{0,2}(\d{4}-\d{2}-\d{2})", head)
    if not m:
        f.err(
            "status-doc: docs/project_status.md has no 'Status date: YYYY-MM-DD' line near the top"
        )
        return None
    return m.group(1)


def check_status_fresh(f: Findings, sdate: str | None) -> None:
    if sdate is None or not (ROOT / ".git").exists():
        return
    status_rel = rel(STATUS_DOC)
    # Working-tree changes (staged + unstaged + untracked) under docs/, CLAUDE.md.
    porcelain = git("status", "--porcelain", "--", "docs", "CLAUDE.md")
    changed = {
        line[3:].strip().split(" -> ")[-1]
        for line in porcelain.splitlines()
        if line.strip()
    }
    status_touched = status_rel in changed
    changed.discard(status_rel)
    # Committed after the status date but status doc not re-dated.
    committed = git(
        "log",
        f"--since={sdate}T23:59:59",
        "--name-only",
        "--format=",
        "--",
        "docs",
        "CLAUDE.md",
    )
    committed_set = {c.strip() for c in committed.splitlines() if c.strip()}
    committed_set.discard(status_rel)
    # Memory dir is outside the repo: compare mtimes to the status date.
    mem_changed: list[str] = []
    if MEMORY_DIR.exists():
        import datetime as dt

        cutoff = dt.datetime.fromisoformat(sdate + "T23:59:59").timestamp()
        for m in MEMORY_DIR.glob("*.md"):
            if m.stat().st_mtime > cutoff:
                mem_changed.append(m.name)
    if (changed or committed_set or mem_changed) and not status_touched:
        detail = sorted(changed | committed_set)[:8]
        mem = sorted(mem_changed)[:5]
        f.err(
            "status-fresh: records changed after project_status.md's status date "
            f"({sdate}) but project_status.md was not touched — update §1/§2/§5 (or bump the "
            "status date with a '(reviewed, no change)' note). Changed: "
            + ", ".join(detail + [f"memory/{m}" for m in mem])
            + (" …" if len(changed | committed_set) > 8 or len(mem_changed) > 5 else "")
        )


def check_phrases(f: Findings, files: list[Path], label: str) -> None:
    for p in files:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for name, trig, qual, msg in PHRASE_RULES:
            tr = re.compile(trig, re.IGNORECASE)
            qr = re.compile(qual, re.IGNORECASE)
            for i, line in enumerate(lines):
                if not tr.search(line):
                    continue
                window = "\n".join(lines[max(0, i - 4) : i + 5])
                if qr.search(window):
                    continue
                f.err(f"phrase[{name}]: {label}{rel(p)}:{i + 1}: {msg}")


def check_refs(f: Findings, files: list[Path], label: str) -> None:
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in REF_RE.finditer(text):
            if not (DOCS / m.group(1)).exists():
                line = text.count("\n", 0, m.start()) + 1
                line_text = text.splitlines()[line - 1]
                if REF_ABSENT_OK.search(line_text):
                    continue
                f.err(
                    f"refs: {label}{rel(p)}:{line}: `docs/{m.group(1)}` does not exist"
                )


def check_sections(f: Findings, docs: list[Path]) -> None:
    for p in docs:
        nums = SECTION_RE.findall(p.read_text(encoding="utf-8"))
        dups = sorted({n for n in nums if nums.count(n) > 1})
        if dups:
            f.err(
                f"sections: {rel(p)} has duplicated section numbers: {', '.join(dups)}"
            )


def check_memory(f: Findings) -> list[Path]:
    if not MEMORY_DIR.exists():
        f.warn(f"memory: {MEMORY_DIR} not found — memory checks skipped")
        return []
    index = MEMORY_DIR / "MEMORY.md"
    files = sorted(p for p in MEMORY_DIR.glob("*.md") if p.name != "MEMORY.md")
    if not index.exists():
        f.err("memory: MEMORY.md index missing")
        return files
    text = index.read_text(encoding="utf-8")
    linked = re.findall(r"\]\(([A-Za-z0-9_\-]+\.md)\)", text)
    counts = {n: linked.count(n) for n in set(linked)}
    for p in files:
        c = counts.get(p.name, 0)
        if c != 1:
            f.err(f"memory: {p.name} is indexed {c}× in MEMORY.md (need exactly 1)")
    for n in counts:
        if not (MEMORY_DIR / n).exists():
            f.err(f"memory: MEMORY.md links to missing file {n}")
    # description lines must not carry an unqualified superseded phrase
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines()[:8]:
            if line.startswith("description:"):
                for name, trig, qual, msg in PHRASE_RULES:
                    if re.search(trig, line, re.IGNORECASE) and not re.search(
                        qual, line, re.IGNORECASE
                    ):
                        f.err(f"memory-desc[{name}]: {p.name}: {msg}")
    return files


def check_claude_md(f: Findings, sdate: str | None) -> None:
    if not CLAUDE_MD.exists():
        f.err("claude-md: CLAUDE.md missing")
        return
    m = re.search(
        r"### Current status \((\d{4}-\d{2}-\d{2})\)",
        CLAUDE_MD.read_text(encoding="utf-8"),
    )
    if not m:
        f.err("claude-md: CLAUDE.md has no '### Current status (YYYY-MM-DD)' heading")
        return
    if sdate and m.group(1) < sdate:
        f.err(
            f"claude-md: CLAUDE.md current-status date {m.group(1)} is older than "
            f"project_status.md's {sdate} — bring the summary forward"
        )


def run() -> Findings:
    f = Findings()
    docs = sorted(DOCS.glob("*.md")) if DOCS.exists() else []
    if not docs:
        f.err("docs/ has no markdown files")
        return f
    check_banners(f, docs)
    sdate = status_date(f)
    check_status_fresh(f, sdate)
    scripts = sorted((ROOT / "scripts").glob("*.py")) + sorted(
        (ROOT / "diff_voyn").rglob("*.py")
    )
    check_phrases(f, docs + [CLAUDE_MD], "")
    check_phrases(f, scripts, "")
    check_refs(f, docs + [CLAUDE_MD] + scripts, "")
    check_sections(f, docs)
    mem_files = check_memory(f)
    check_refs(f, mem_files, "memory/")
    check_phrases(f, mem_files, "memory/")
    check_claude_md(f, sdate)
    return f


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--hook", choices=["stop", "session-start"], help="emit Claude Code hook JSON"
    )
    args = ap.parse_args()
    f = run()
    if args.hook:
        payload = {}
        try:
            payload = json.loads(sys.stdin.read() or "{}")
        except (json.JSONDecodeError, OSError):
            pass
        report = (
            "\n".join(f"- {e}" for e in f.errors)
            or "documentation coherence check: clean"
        )
        if args.hook == "stop":
            if f.errors and not payload.get("stop_hook_active"):
                print(
                    json.dumps(
                        {
                            "decision": "block",
                            "reason": (
                                "Documentation coherence check failed (scripts/doc_coherence_check.py). "
                                "Fix before finishing — dated records must stay labelled and "
                                "docs/project_status.md must be brought forward:\n"
                                + report
                            ),
                        }
                    )
                )
            elif f.errors:
                # second pass in the same turn: do not loop, just surface
                print(
                    json.dumps(
                        {"systemMessage": "doc coherence still failing:\n" + report}
                    )
                )
            return 0
        # session-start: inform, never block
        if f.errors:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": "Documentation coherence check found drift at session start "
                            "(run `uv run python scripts/doc_coherence_check.py`):\n"
                            + report,
                        }
                    }
                )
            )
        return 0
    for w in f.warnings:
        print(f"WARN  {w}")
    for e in f.errors:
        print(f"ERROR {e}")
    if f.errors:
        print(
            f"\n{len(f.errors)} finding(s). See scripts/doc_coherence_check.py docstring and docs/project_status.md."
        )
        return 1
    print("documentation coherence check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
