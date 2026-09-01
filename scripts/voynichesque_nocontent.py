"""Content-free voynichesque controls — strict negatives for the Phase-6 rule.

The Phase-6 ``voynichesque`` control enciphers REAL held-out text, so it is a
wrong-hypothesis partial decode of real language, not a strict negative (its
token types deterministically encode 1-3-letter plaintext chunks; see the
2026-08-31 oracle study). This battery builds the strict version: for each of
the nine Phase-6 voynichesque instances, the SAME generator seed (hence the
identical parameter draw, cipher alphabets, chunk boundaries and y-decoration
— the RNG stream is consumed identically for an equal-length source) applied
to a LETTER-SHUFFLED copy of the same source window. Unigram statistics and
the full glyph grammar survive; sequential content provably does not.

If these twins reproduce the Phase-6 voynichesque margin band (0.92-1.51 on
the glyph heads, 0.19-0.79 under naibbe), that band was glyph grammar, not
leaked content, and the twins become the legitimate "Voynich-shaped
gibberish" negative for every head.

Result (2026-08-31, analysis/phase6/controls_nocontent/report.json;
docs/voynichesque_nocontent_restart.md "Results"): mixed. The homophonic
band was content-inflated — twin − real = −0.27 bits/char on 27/27 pairs
(real 0.85–1.51 → twin 0.55–1.24); sub1to1 (−0.02) and naibbe (+0.02)
reproduced, i.e. pure glyph grammar. Strict-gibberish ceiling 1.40; 0/66
cells language-like, 9/9 instances abstain; the Phase-6 P0 near-miss
italian/t1 (1.51) falls to 1.14–1.24. The real-text ``voynichesque`` control
is therefore a wrong-hypothesis control, these twins are the strict negative.

Stages: this script only GENERATES (instances + manifest, Phase-6 format).
Solve / score / report reuse scripts/vms_controls.py verbatim:

  uv run python scripts/voynichesque_nocontent.py
  uv run python scripts/vms_controls.py --stage solve  --out-dir DATA_ROOT/analysis/phase6/controls_nocontent
  uv run python scripts/vms_controls.py --stage score  --out-dir ... --shard 0/2   (GPU pair)
  uv run python scripts/vms_controls.py --stage report --out-dir ...

Artifacts: DATA_ROOT/analysis/phase6/controls_nocontent/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.heads.naibbe_parse import NaibbeParser
from diff_voyn.heads.synth import HeldoutSampler
from diff_voyn.vms.controls import (
    _letters_to_text,
    _rng,
    voynichesque_instances,
)

LANGS = ("latin", "italian", "german")
FAM = {"latin": "romance", "italian": "romance", "german": "germanic"}
CONTROL = "voynichesque_nocontent"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "analysis" / "phase6" / "controls_nocontent",
    )
    p.add_argument(
        "--ref-dir",
        type=Path,
        default=root / "analysis" / "phase6" / "controls",
        help="Phase-6 controls dir holding the real-text twins to match",
    )
    p.add_argument("--per-language", type=int, default=3)
    p.add_argument("--length", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="Phase-6 controls seed")
    args = p.parse_args()

    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    parser = NaibbeParser()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(inst: dict):
        fname = re.sub(r"[^A-Za-z0-9_]+", "_", inst["name"]) + f"_{inst['kind']}.json"
        inst["control"] = CONTROL
        (args.out_dir / fname).write_text(json.dumps(inst))
        manifest.append(
            {
                "name": inst["name"],
                "kind": inst["kind"],
                "file": fname,
                "control": CONTROL,
                "truth": {
                    k: v
                    for k, v in inst["truth"].items()
                    if k not in ("plain_ids", "sym_to_letter")
                },
                "coverage": inst["coverage"],
                "n_symbols": inst["n_symbols"],
                "n_stream": inst["n_stream"],
            }
        )
        print(f"  {inst['name']:34s} {inst['kind']:5s} n_stream {inst['n_stream']}")

    for lang in LANGS:
        sampler = HeldoutSampler(corpus_dir, splits, lang)
        for t in range(args.per_language):
            # Phase-6 seed path, replayed exactly (vms/controls.py build_controls)
            rng = _rng("voynichesque", args.seed, lang, t)
            src = sampler.sample(int(args.length * 0.75), rng)
            gseed = int(rng.integers(2**31))

            # verify the replay against the stored real-text instance
            ref_path = args.ref_dir / f"voynichesque_{lang}_t{t}_eva.json"
            ref = json.loads(ref_path.read_text())
            real = voynichesque_instances(
                "check", _letters_to_text(src), gseed, parser, {}
            )[0]
            stream_real = "".join(real["alphabet"][s] for s in real["symbols"])
            stream_ref = "".join(ref["alphabet"][s] for s in ref["symbols"])
            assert stream_real == stream_ref, f"replay mismatch on {lang}/t{t}"

            # the content-free twin: same seed, letter-shuffled source
            shuf = _rng("voynichesque-nocontent", args.seed, lang, t).permutation(src)
            truth = {
                "language": lang,
                "family": FAM[lang],
                "in_inventory": True,
                "source_language": lang,
                "source": "letter-shuffled",
                "twin_of": f"voynichesque/{lang}/t{t}",
            }
            for inst in voynichesque_instances(
                f"voynichesque_nc/{lang}/t{t}",
                _letters_to_text(shuf),
                gseed,
                parser,
                truth,
            ):
                if inst["kind"] == "eva":
                    # identical parameter draw => identical token count and
                    # per-token glyph lengths
                    # identical parameter draw => identical chunk count (the
                    # character stream length differs: glyph lengths depend
                    # on which letters occur)
                    assert inst["truth"]["n_tokens"] == ref["truth"]["n_tokens"], (
                        f"chunking mismatch on {lang}/t{t}: "
                        f"{inst['truth']['n_tokens']} vs {ref['truth']['n_tokens']}"
                    )
                emit(inst)

    write_json_atomic(args.out_dir / "manifest.json", manifest)
    print(len(manifest), "instances ->", args.out_dir)


if __name__ == "__main__":
    main()
