---
name: coherence-review
description: Semantic documentation-coherence review for diff-voyn — run after any study that changes a result, a solver, a judge or an "open item". Re-surveys docs + memory for statements superseded by newer results, updates docs/project_status.md (§1/§2/§5/§6), annotates the superseded sentences in place, and rewrites affected memory descriptions. The mechanical half (banners, phrase qualifiers, refs, index) is scripts/doc_coherence_check.py, run automatically by the Stop hook.
---

# Coherence review (semantic half)

`scripts/doc_coherence_check.py` (Stop / SessionStart / pre-commit hooks) catches the
*mechanical* drift — a doc without a banner, a known-superseded phrase without its
qualifier, a dangling `docs/…` pointer, an unindexed memory, docs edited without touching
`docs/project_status.md`. It cannot tell that a **new result supersedes an old claim**.
That is this skill's job. Run it after a study lands, or when `/coherence-review` is typed.

## Procedure

1. **Anchor.** Read `docs/project_status.md` §1 (current status) and §5 (superseded
   statements). Read the study's own doc/section and its memory file. Write down, in one
   list, every *fact that changed*: a number, a "solver of record", a judge property, an
   item moving from open → done, a claim refuted.
2. **Sweep.** For each changed fact, grep the old value / old wording across `docs/*.md`,
   `CLAUDE.md`, `~/.claude/projects/-workspace/memory/*.md`, and script docstrings. If the
   sweep is wide (> ~10 files), fan out read-only survey agents by slice (early phases;
   Phase 5/6 + overview; wordhom/alt-loop; judge/controls + docstrings; memory) with the
   brief pattern used on 2026-09-01: per-file card, dated timeline entries, stale
   statements with `file:line` + suggested fix type (banner / inline note / rewrite / delete).
3. **Arbitrate.** Add or amend the entry in `project_status.md` §5 (canonical reading, date,
   record), update §1 and §2 (ladder row), move items in §6, and bump the `Status date`.
   `CLAUDE.md` "Current status (…)" block: update the sentence and the date.
4. **Annotate in place.** Never delete history. After each superseded sentence add
   `*[Superseded YYYY-MM-DD: … — see docs/…, docs/project_status.md §5.N.]*`; outright
   errors get rewritten with `*(corrected YYYY-MM-DD; originally read "…")*`. Extend the
   banner of any doc whose *headline* is now stale.
5. **Memory.** Rewrite the `description:` line of every memory whose hook quotes the old
   value (recall reads descriptions); append a `**Status (YYYY-MM-DD):**` paragraph; update
   the MEMORY.md hook. New study → new memory + index line in the right group.
6. **Teach the checker.** If the superseded claim has a recognisable phrasing, add a
   `PHRASE_RULES` entry (trigger, qualifier, message pointing at §5.N) so the mechanical
   check catches recurrences; add a `sections`/`refs` exception only when justified.
7. **Verify.** `uv run python scripts/doc_coherence_check.py` must be clean; grep once
   more for the old value with no qualifier; run the touched tests if docstrings changed.
   Report: facts changed, files annotated, §5 entries added, memories rewritten.

## Don'ts
- Don't update numbers silently inside a dated record — label them.
- Don't quote a wall/SER/margin without its solver stage and date.
- Don't put status content in `MEMORY.md` (pointers only) or edit `reference_docs/`
  (pre-registration record; corrections live in `project_status.md` §5).
