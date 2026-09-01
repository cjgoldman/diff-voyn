"""Naibbe-ciphertext twins for the wordhom head — short test (2026-08-31).

Question: how much of the structure a wordhom decode finds on Naibbe
ciphertext is CONTENT vs Naibbe's token grammar? Naibbe is itself a
word-homophonic cipher in the wordhom sense (each cipher word type
deterministically encodes one 1-2-letter plaintext segment from the pinned
tables), so the wordhom key space partially represents its true key — the
strongest representability of any control generator. This builds ONE
seed-paired twin set:

  naibbetwin/german/Alike      Naibbe on a real held-out German window
  naibbetwin_nc/german/Alike   Naibbe (same cipher seed) on the SAME window
                               letter-shuffled — strict negative

Both are emitted into the wordhom battery dir (control tag ``naibbe_twin``)
so the standard machinery runs unchanged:

  uv run python scripts/naibbe_wordhom_twins.py                      # build
  uv run python scripts/wordhom_battery.py --stage solve --only naibbetwin --hyps german --workers 12
  # then per cell (solver convention: wildcard -> anneal, CLAUDE.md):
  #   altloop_pol.py --battery --cells naibbetwin/german/Alike:german --wild --tag _ntw_wild ...
  #   altloop_pol.py ... --wild --wild-anneal 0,40 --start-from _ntw_wild --tag _ntw_anneal
  #   judge_at_ser.py --battery <cell> --run-tags _ntw_wild _ntw_anneal --tag _naibbe_twin

Truth stored per instance: ``plain_ids`` (the pre-mapped plaintext the
segments spell — the judge computes letter SER against it for every key),
``segments`` (per-token 1-2-letter chunks) for oracle/overlap analysis, and
``kind: "naibbe"`` so no machinery mistakes it for a wordhom-cipher truth key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from diff_voyn.ciphers.external import data_root
from diff_voyn.ciphers.naibbe import NaibbeCipher
from diff_voyn.corpus.splits import load_splits
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.ngram import LETTER_TO_IDX
from diff_voyn.heads.synth import HeldoutSampler
from diff_voyn.vms.apply import build_ngram_evaluator
from diff_voyn.vms.controls import _letters_to_text
from diff_voyn.vms.presentations import wordtypes_presentation


def build_instance(name: str, plain_letters: np.ndarray, cipher_seed: int, truth: dict):
    tokens, segments = NaibbeCipher(seed=cipher_seed).encipher(
        _letters_to_text(plain_letters)
    )
    plain_ids = [LETTER_TO_IDX[c] for c in "".join(segments)]
    # type determinism: does each token type encode exactly one segment?
    per_type = defaultdict(Counter)
    for w, s in zip(tokens, segments):
        per_type[w][s] += 1
    ambiguous = sum(len(c) > 1 for c in per_type.values())
    off_major = sum(sum(c.values()) - c.most_common(1)[0][1] for c in per_type.values())
    seg_len = np.array([len(s) for s in segments])
    pres = wordtypes_presentation("ctrl", "-", None, words=tokens, name=name)
    print(
        f"{name}: {len(tokens)} tokens, {pres.n_symbols} types "
        f"({len(tokens)/pres.n_symbols:.1f}/type), plaintext {len(plain_ids)} letters; "
        f"segments 1-letter {np.mean(seg_len==1):.2f} / 2-letter {np.mean(seg_len==2):.2f} of tokens; "
        f"type ambiguity {ambiguous}/{pres.n_symbols} types, {off_major}/{len(tokens)} tokens off-majority"
    )
    return {
        "name": name,
        "kind": pres.kind,
        "n_symbols": pres.n_symbols,
        "alphabet": pres.alphabet,
        "n_stream": len(pres.symbols),
        "coverage": pres.coverage,
        "symbols": pres.symbols.tolist(),
        "token_pos": pres.token_starts.tolist(),
        "all_tokens": tokens,
        "truth": dict(
            truth,
            kind="naibbe",
            cipher_seed=cipher_seed,
            n_tokens=len(tokens),
            plain_ids=plain_ids,
            segments=segments,
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=14000, help="source letters (A-like)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cipher-seed", type=int, default=7)
    args = p.parse_args()

    import wordhom_battery as wb

    root = data_root()
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    ev = build_ngram_evaluator()
    smp = HeldoutSampler(corpus_dir, splits, "german")
    rng = np.random.default_rng(args.seed + 1717)
    plain = wb.sample_long(smp, args.length, rng, ev.lms["german"])
    shuf = np.random.default_rng(args.seed + 2727).permutation(plain)

    out_dir = wb.battery_dir(root)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    manifest = [m for m in manifest if not m["name"].startswith("naibbetwin")]

    base = {"language": "german", "family": "germanic", "in_inventory": True}
    insts = [
        build_instance(
            "naibbetwin/german/Alike",
            plain,
            args.cipher_seed,
            dict(base, source="heldout", twin_of="naibbetwin_nc/german/Alike"),
        ),
        build_instance(
            "naibbetwin_nc/german/Alike",
            shuf,
            args.cipher_seed,
            dict(base, source="letter-shuffled", twin_of="naibbetwin/german/Alike"),
        ),
    ]
    for inst in insts:
        fname = re.sub(r"[^A-Za-z0-9_]+", "_", inst["name"]) + f"_{inst['kind']}.json"
        inst["control"] = "naibbe_twin"
        (out_dir / fname).write_text(json.dumps(inst))
        manifest.append(
            {
                "name": inst["name"],
                "kind": inst["kind"],
                "file": fname,
                "control": "naibbe_twin",
                "truth": {
                    k: v
                    for k, v in inst["truth"].items()
                    if k not in ("plain_ids", "segments")
                },
                "coverage": inst["coverage"],
                "n_symbols": inst["n_symbols"],
                "n_stream": inst["n_stream"],
            }
        )
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(f"2 instances -> {out_dir} (manifest now {len(manifest)} entries)")


if __name__ == "__main__":
    main()
