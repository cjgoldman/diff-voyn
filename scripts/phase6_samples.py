"""Phase 6, task 6.7 support — representative decode samples for the write-up.

The 6.2 verdict is a scalar (0 of 87 cells language-like); this script shows
the reader WHAT the instrument judged. For each head it takes the cell with
the lowest full-stream calibrated plaintext bits (the decode that came
closest to the clean-text band — the most charitable sample the manuscript
produces), regenerates the full-stream decode from the final key stored in
the score record, and renders an aligned (ciphertext, decode) excerpt with
the manuscript's word boundaries. Two contrasts frame the samples:

- a positive control (synthetic sub1to1 Latin) through the identical solve
  path, so the reader sees what a TRUE decipherment looks like at the same
  point in the pipeline;
- the letter-shuffled copy of the best VMS decode — the other side of the
  structure margin, i.e. exactly what the abstention rule compares against.

Excerpts are deterministic (stream start, plus one mid-stream excerpt for
the headline cell); nothing is re-scored and nothing is cherry-picked.

Usage:
  uv run python scripts/phase6_samples.py            # -> analysis/phase6/samples.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.ngram import LETTERS
from diff_voyn.vms.apply import KEY, arithmetic_stream, build_ngram_evaluator
from diff_voyn.vms.presentations import dialect_words

EXCERPT_CHARS = 300
EXCERPT_TOKENS = 40


def load_scores(out_dir: Path) -> dict[tuple, dict]:
    recs = {}
    for p in sorted(out_dir.glob("vms_scores*.json")):
        for r in json.loads(p.read_text())["instances"]:
            recs[tuple(r[k] for k in KEY)] = r
    return recs


def letters_str(ids) -> str:
    return "".join(LETTERS[i] for i in np.asarray(ids, dtype=np.int64))


def group_words(chars: str, lengths: list[int], sep: str = ".") -> str:
    """Insert word separators into an unspaced stream, given word lengths."""
    out, pos = [], 0
    for n in lengths:
        out.append(chars[pos : pos + n])
        pos += n
    assert pos == len(chars)
    return sep.join(out)


def take_words(lengths: list[int], start_char: int, n_chars: int) -> tuple[int, int]:
    """(first word index, n words) covering [start_char, start_char+n_chars),
    snapped to word boundaries."""
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    w0 = int(np.searchsorted(bounds, start_char, side="right") - 1)
    w1 = int(np.searchsorted(bounds, start_char + n_chars, side="left"))
    return w0, max(w1 - w0, 1)


def eva_word_lengths(inst: dict, root: Path) -> list[int]:
    """Word lengths of the model-facing stream. EVA instances re-derive them
    from the IVTFF source (the stream is the concatenation of the dialect's
    words); Boxer symbol instances carry token_starts."""
    if inst["transcription"] == "boxer":
        starts = np.asarray(inst["token_starts"])
        ends = np.concatenate([starts[1:], [len(inst["symbols"])]])
        return (ends - starts).tolist()
    words = dialect_words(inst["transcription"], root)[inst["dialect"]]
    stream = "".join(inst["alphabet"][s] for s in inst["symbols"])
    assert "".join(words) == stream, "word list does not reproduce the stream"
    return [len(w) for w in words]


def symbol_excerpt(inst: dict, lengths: list[int], w0: int, n_words: int) -> str:
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    a, b = int(bounds[w0]), int(bounds[w0 + n_words])
    chars = "".join(inst["alphabet"][s] for s in inst["symbols"][a:b])
    return group_words(chars, lengths[w0 : w0 + n_words])


def cell_caption(cell: dict) -> dict:
    return {
        k: cell[k]
        for k in (
            "instance",
            "presentation",
            "head",
            "window",
            "hypothesis",
            "final_source",
            "plain_bits",
            "plain_bits_sem",
            "structure_margin",
            "total_per_all_symbols",
            "no_cipher_baseline_bits_per_symbol",
            "coverage",
            "language_like",
        )
    }


def sample_substitution(cell, rec, inst, root, *, mid_excerpt=False) -> dict:
    """sub1to1 / homophonic on a symbol stream: 1:1 char alignment."""
    m = np.asarray(rec["final"]["key"]["map"], dtype=np.int64)
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    dec = letters_str(m[sym])
    lengths = eva_word_lengths(inst, root)
    excerpts = []
    starts = [0] + ([len(dec) // 2] if mid_excerpt else [])
    for sc in starts:
        w0, nw = take_words(lengths, sc, EXCERPT_CHARS)
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        a, b = int(bounds[w0]), int(bounds[w0 + nw])
        excerpts.append(
            {
                "position": f"chars {a}-{b} of {len(dec)}",
                "ciphertext": symbol_excerpt(inst, lengths, w0, nw),
                "decode": group_words(dec[a:b], lengths[w0 : w0 + nw]),
            }
        )
    return {
        "cell": cell_caption(cell),
        "alignment": "1 letter per symbol",
        "excerpts": excerpts,
    }


def sample_naibbe(cell, rec, inst, ng, parser) -> dict:
    """Naibbe: decode with per-token branch info (1 letter per unigram token,
    2 per bigram token)."""
    from diff_voyn.heads.rung3_naibbe import NaibbeBlockHead

    h = NaibbeBlockHead(ng, parser, seed=0)
    maps = {
        tuple(k.split("|")): np.asarray(v, dtype=np.int64)
        for k, v in rec["final"]["key"]["maps"].items()
    }
    tokens = inst["tokens"]
    parses = parser.parse_stream(tokens)
    letters, branches, _ = h.decode(parses, maps, rec["hypothesis"])
    dec = letters_str(letters)
    # letters per token from the chosen branch: unigram -> 1, bigram -> 2
    lens = []
    for p, bi in zip(parses, branches):
        n_uni = 1 if p.uni is not None else 0
        lens.append(1 if (p.uni is not None and bi == 0) else 2 if p.bi else n_uni)
    assert sum(lens) == len(dec)
    nt = EXCERPT_TOKENS
    bounds = np.concatenate([[0], np.cumsum(lens)])
    return {
        "cell": cell_caption(cell),
        "alignment": "1 letter per unigram token, 2 per bigram token; "
        "only Naibbe-parseable words (73-82% of words) appear",
        "excerpts": [
            {
                "position": f"tokens 0-{nt} of {len(tokens)}",
                "ciphertext": ".".join(tokens[:nt]),
                "decode": group_words(dec[: bounds[nt]], lens[:nt]),
            }
        ],
    }


def sample_arithmetic(cell, rec, inst, ng) -> dict:
    """Arithmetic (segmented): one letter per kept token."""
    from diff_voyn.heads.rung4_arithmetic import (
        ArithmeticHead,
        segmented_admissible_mask,
    )

    ids, starts = arithmetic_stream(inst)
    h = ArithmeticHead(ng, seed=0)
    adm = segmented_admissible_mask(ids, starts)
    _, letters, _ = h.decode_segmented(
        ids,
        adm,
        np.asarray(rec["final"]["key"]["v"]),
        np.asarray(rec["final"]["key"]["u"]),
        language=rec["hypothesis"],
    )
    dec = letters_str(letters)
    ends = np.concatenate([starts[1:], [len(ids)]])
    toks = [
        "".join(inst["alphabet"][g] for g in ids[s:e]) for s, e in zip(starts, ends)
    ]
    assert len(dec) == len(toks), "segmented decode is one letter per kept token"
    nt = EXCERPT_TOKENS
    return {
        "cell": cell_caption(cell),
        "alignment": "1 letter per kept Boxer token (length 2-6 only); "
        "glyphs in Boxer's notation",
        "excerpts": [
            {
                "position": f"tokens 0-{nt} of {len(toks)}",
                "ciphertext": ".".join(toks[:nt]),
                "decode": ".".join(dec[:nt]),
            }
        ],
    }


def positive_control_sample(controls_dir: Path) -> dict:
    """The same solve path on a synthetic sub1to1 Latin positive: what a
    true decipherment looks like, with letter accuracy against the truth."""
    inst = json.loads((controls_dir / "positive_latin_t0_eva.json").read_text())
    solves = json.loads((controls_dir / "solves.json").read_text())["instances"]
    rec = next(
        r
        for r in solves
        if r["instance"] == "positive/latin/t0"
        and r["head"] == "sub1to1"
        and r["hypothesis"] == "latin"
    )
    best = max(rec["candidates"], key=lambda c: c["inner"])
    m = np.asarray(best["map"], dtype=np.int64)
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    dec, truth = m[sym], np.asarray(inst["truth"]["plain_ids"], dtype=np.int64)
    acc = float((dec == truth).mean())
    return {
        "instance": "positive/latin/t0 (synthetic 1:1, 2000 chars)",
        "head": "sub1to1",
        "hypothesis": "latin",
        "letter_accuracy_vs_truth": acc,
        "decode_excerpt": letters_str(dec[:EXCERPT_CHARS]),
        "note": "inner-search best candidate, identical solve path; controls "
        "streams are whitespace-stripped like every model-facing stream",
    }


def fmt_md(samples: dict) -> str:
    md = [
        "### Representative decodes behind the 6.2 abstention (task 6.7)",
        "",
        "What the frozen evaluator was judging, not just the scalar: for each",
        "head, the cell with the LOWEST full-stream plaintext bits — the most",
        "language-like decode the manuscript produces anywhere in the 87-cell",
        "grid. `.` marks the manuscript's word boundaries (the evaluator sees",
        "the unspaced stream).",
        "",
    ]
    for s in samples["vms_cells"]:
        c = s["cell"]
        md += [
            (
                f"#### {c['head']} — {c['instance']} ({c['presentation']}), "
                f"{c['hypothesis']} hypothesis"
            ),
            "",
            (
                f"plaintext {c['plain_bits']:.3f} ± {c['plain_bits_sem']:.3f} bits/char, "
                f"structure margin {c['structure_margin']:.2f} (rule needs ≥ 1.5), "
                f"MDL total {c['total_per_all_symbols']:.3f} bits/symbol "
                f"(no-cipher baseline {c['no_cipher_baseline_bits_per_symbol']:.3f}), "
                f"coverage {c['coverage']:.2f} — "
                f"{'language-like' if c['language_like'] else 'NOT language-like'}. "
                f"Alignment: {s['alignment']}."
            ),
            "",
        ]
        for e in s["excerpts"]:
            md += [
                f"*{e['position']}*",
                "```",
                "cipher: " + e["ciphertext"],
                "decode: " + e["decode"],
                "```",
                "",
            ]
    p = samples["positive_control"]
    md += [
        "#### Contrast 1 — a true decipherment through the same pipeline",
        "",
        f"{p['instance']}, {p['head']} head, {p['hypothesis']} hypothesis; "
        f"letter accuracy vs truth {p['letter_accuracy_vs_truth']:.3f}. "
        + p["note"]
        + ".",
        "```",
        p["decode_excerpt"],
        "```",
        "",
        "#### Contrast 2 — the shuffled side of the structure margin",
        "",
        (
            "The same best-cell decode excerpt, letter-shuffled (the per-window "
            "control every cell is paired with). The abstention rule asks the "
            "decode to beat THIS by ≥ 1.5 bits/char; the best VMS cell beats it "
            f"by {samples['shuffled_contrast']['structure_margin']:.2f}."
        ),
        "```",
        "decode:   " + samples["shuffled_contrast"]["decode"],
        "shuffled: " + samples["shuffled_contrast"]["shuffled"],
        "```",
    ]
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument("--out-dir", type=Path, default=root / "analysis" / "phase6")
    p.add_argument(
        "--pres-dir", type=Path, default=root / "analysis" / "phase6" / "presentations"
    )
    args = p.parse_args()

    report = json.loads((args.out_dir / "vms_report.json").read_text())
    scores = load_scores(args.out_dir)
    best = {}
    for c in report["cells"]:
        if c["head"] not in best or c["plain_bits"] < best[c["head"]]["plain_bits"]:
            best[c["head"]] = c

    ng = build_ngram_evaluator()
    parser = None
    if "naibbe" in best:
        from diff_voyn.heads.naibbe_parse import NaibbeParser

        parser = NaibbeParser()
        parser.build_blocks()

    vms_cells = []
    headline = min(best.values(), key=lambda c: c["plain_bits"])
    for head in ("homophonic", "sub1to1", "naibbe", "arithmetic"):
        if head not in best:
            continue
        cell = best[head]
        rec = scores[tuple(cell[k] for k in KEY)]
        tr, d = cell["instance"].split("/")
        stem = f"{tr}_{d}_{cell['presentation']}"
        inst = json.loads((args.pres_dir / f"{stem}.json").read_text())
        if head in ("sub1to1", "homophonic"):
            s = sample_substitution(cell, rec, inst, root, mid_excerpt=cell is headline)
        elif head == "naibbe":
            s = sample_naibbe(cell, rec, inst, ng, parser)
        else:
            s = sample_arithmetic(cell, rec, inst, ng)
        vms_cells.append(s)
        print(f"{head}: {cell['instance']} {cell['hypothesis']} done", flush=True)

    head_sample = next(s for s in vms_cells if s["cell"]["head"] == headline["head"])
    exc = head_sample["excerpts"][0]
    dec_plain = exc["decode"].replace(".", "")
    shuf = "".join(np.random.default_rng(0).permutation(list(dec_plain)))
    samples = {
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_rule": "per head, the cell with the lowest full-stream "
        "calibrated plaintext bits over all 87 cells; excerpts at fixed, "
        "deterministic positions",
        "vms_cells": vms_cells,
        "positive_control": positive_control_sample(args.out_dir / "controls"),
        "shuffled_contrast": {
            "cell": head_sample["cell"],
            "decode": dec_plain,
            "shuffled": shuf,
            "structure_margin": headline["structure_margin"],
        },
    }
    write_json_atomic(args.out_dir / "samples.json", samples)
    md = fmt_md(samples)
    (args.out_dir / "samples.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
