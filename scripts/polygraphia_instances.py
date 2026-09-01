"""Polygraphia anchor instances for the word-homophonic head
(docs/polygraphia_digitization_scope.md §6).

Enciphers held-out plaintext of each inventory language through the
digitized Polygraphia tables (deterministic column cipher, start column 1)
and writes wordtypes instances in the wordhom-control format, so
`scripts/wordhom_study.py`-style solve/score stages can consume them.

Two shapes per language, fixed by the findability contrast of §6.2:
- ``cyclic``: length ≫ n_columns → tokens/type well above the ≥8 wall;
- ``hapax``: length = n_columns → every column used once, tokens/type ≈ 1.

Because the same word can stand under different letters in different
columns, a type-deterministic (wordhom) key cannot represent the cipher
exactly; each instance records the ORACLE type-key ceiling — the fraction
of tokens correct under the best possible type→letter map ("oracle_type_acc")
— and per-type majority units in ``truth.sym_to_unit``.

Usage: uv run python scripts/polygraphia_instances.py \
  --tables DATA_ROOT/external/polygraphia/provisional_tables.csv --n-columns 41
"""

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import data_root
from diff_voyn.ciphers.polygraphia import (
    PolygraphiaCipher,
    PolygraphiaTables,
    polygraphia_pre_map,
)
from diff_voyn.vocab import LETTERS

LANGS = ("latin", "italian", "german")


def build_instance(name, plain_ids, cipher, truth, n_top=None):
    from diff_voyn.vms.presentations import wordtypes_presentation

    text = "".join(LETTERS[i] for i in plain_ids)
    text = polygraphia_pre_map(text)
    plain_ids = np.array([LETTERS.index(c) for c in text], dtype=np.int64)
    words = cipher.encipher(text, start_column=1)
    cands = cipher.decipher_candidates(words, start_column=1)
    assert all(ch in cs for ch, cs in zip(text, cands))
    n_ambig = sum(1 for cs in cands if len(cs) > 1)

    pres = wordtypes_presentation("ctrl", "-", n_top, words=words, name=name)
    # majority truth unit per presented type + oracle ceiling
    per_type = defaultdict(Counter)
    for w, i in zip(words, plain_ids):
        per_type[w][int(i)] += 1
    majority = {w: c.most_common(1)[0][0] for w, c in per_type.items()}
    oracle_correct = sum(c.most_common(1)[0][1] for c in per_type.values())
    colliding = sum(1 for c in per_type.values() if len(c) > 1)

    rec = {
        "name": name,
        "kind": pres.kind,
        "n_symbols": pres.n_symbols,
        "alphabet": pres.alphabet,
        "n_stream": len(pres.symbols),
        "coverage": pres.coverage,
        "symbols": pres.symbols.tolist(),
        "token_pos": pres.token_starts.tolist(),
        "all_tokens": words,
        "truth": dict(
            truth,
            kind="wordhom",  # downstream tooling keys on this
            cipher="polygraphia",
            plain_ids=plain_ids.tolist(),
            unit_ids=plain_ids.tolist(),  # every unit is a single letter
            hyp_bigrams=[],  # letter-only hypothesis space (no doubled units)
            sym_to_unit=[majority[w] for w in pres.alphabet],
            n_types=len(per_type),
            start_column=1,
            n_columns=len(cipher.tables),
            oracle_type_acc=oracle_correct / len(words),
            colliding_types=colliding,
            ambiguous_positions=n_ambig,
        ),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None, help="tables CSV (default: frozen v1)")
    ap.add_argument("--n-columns", type=int, default=None)
    ap.add_argument("--book", default="b1")
    ap.add_argument("--cyclic-length", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from diff_voyn.corpus.splits import load_splits
    from diff_voyn.heads.synth import HeldoutSampler

    tables = (
        PolygraphiaTables(args.tables or None, book=args.book, n_columns=args.n_columns)
        if args.tables
        else PolygraphiaTables(book=args.book, n_columns=args.n_columns)
    )
    cipher = PolygraphiaCipher(tables)
    out_dir = pathlib.Path(
        args.out_dir or data_root() / "analysis" / "polygraphia_anchor" / "instances"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = data_root() / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    manifest = []
    for lang in LANGS:
        fam = {"latin": "romance", "italian": "romance", "german": "germanic"}[lang]
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        rng = np.random.default_rng(args.seed)
        for tag, length in (("cyclic", args.cyclic_length), ("hapax", len(tables))):
            plain = sampler.sample(length, rng)
            truth = {
                "language": lang,
                "family": fam,
                "in_inventory": True,
                "shape": tag,
            }
            inst = build_instance(
                f"polygraphia/{lang}/{tag}", np.asarray(plain), cipher, truth
            )
            fname = f"polygraphia_{lang}_{tag}.json"
            (out_dir / fname).write_text(json.dumps(inst))
            m = {
                "name": inst["name"],
                "file": fname,
                "language": lang,
                "shape": tag,
                "n_stream": inst["n_stream"],
                "n_types": inst["truth"]["n_types"],
                "tokens_per_type": inst["n_stream"] / inst["truth"]["n_types"],
                "oracle_type_acc": inst["truth"]["oracle_type_acc"],
                "colliding_types": inst["truth"]["colliding_types"],
            }
            manifest.append(m)
            print(json.dumps(m))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} instances -> {out_dir}")


if __name__ == "__main__":
    main()
