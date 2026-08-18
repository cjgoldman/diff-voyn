"""Cipher generator acceptance run (task 0.7).

- Naibbe (pinned df3d074): generates aligned pairs on sample text per language.
- Arithmetic (voynpy.pseudo_vms, pinned): custom alphabet = frozen vocab,
  default parameters, fixed seed; encode→decode round-trip must be exact;
  ``doubling_strength`` tuned per language to the VMS ~0.92% doubling rate;
  tuned tables persisted to DATA_ROOT/ciphers/ (persisted-determinism).
- Voynichesque negative control: generates.

Outputs DATA_ROOT/ciphers/acceptance_stats.json with entropy, token-length
distribution and doubling rates per language.

Run: ``uv run python scripts/tune_ciphers.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.arithmetic import VMS_DOUBLING_RATE, ArithmeticCipher
from diff_voyn.ciphers.controls import Voynichesque
from diff_voyn.ciphers.external import data_root
from diff_voyn.ciphers.naibbe import NaibbeCipher
from diff_voyn.corpus.splits import load_splits

SAMPLE_CHARS = 60_000
DOUBLING_TOLERANCE = 0.004  # |achieved - 0.0092| must be under this


def language_sample(corpus_dir: Path, lang: str, splits: dict) -> str:
    doc_id = splits["languages"][lang]["train"][0]["doc_id"]
    return (corpus_dir / lang / "docs" / f"{doc_id}.txt").read_text()[:SAMPLE_CHARS]


def main() -> None:
    root = data_root()
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    out_dir = root / "ciphers"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": 42, "target_doubling_rate": VMS_DOUBLING_RATE}
    failures = []

    naibbe = NaibbeCipher(seed=0)
    voyn = Voynichesque()

    for lang in ("latin", "italian", "german"):
        sample = language_sample(corpus_dir, lang, splits)
        entry: dict = {}

        tokens, segments = naibbe.encipher(sample[:5000])
        entry["naibbe"] = {
            "tokens": len(tokens),
            "plaintext_letters": sum(len(s) for s in segments),
            "expansion_tokens_per_letter": len(tokens) / sum(len(s) for s in segments),
        }

        arith = ArithmeticCipher()  # upstream defaults, seed 42
        tuned = arith.tune_doubling(sample)
        cipher = arith.encode(sample)
        decoded = arith.decode_text(cipher)
        # decode returns exactly the alphabet-covered input (spaces dropped)
        expected = "".join(c for c in sample if c in arith.enc.alphabet)
        roundtrip_ok = decoded == expected
        stats = arith.stats(cipher)
        doubling_ok = (
            abs(stats["doubling_rate"] - VMS_DOUBLING_RATE) < DOUBLING_TOLERANCE
        )
        if not roundtrip_ok:
            failures.append(f"{lang}: arithmetic round-trip")
        if not doubling_ok:
            failures.append(f"{lang}: doubling {stats['doubling_rate']:.4%}")
        table_path = out_dir / f"pseudo_vms_{lang}.csv"
        arith.save(table_path)
        entry["arithmetic"] = {
            "tuned_doubling_strength": tuned,
            "round_trip_exact": roundtrip_ok,
            "table_path": str(table_path),
            **stats,
        }

        control = voyn.generate(sample[:5000], seed=0)
        entry["voynichesque_control"] = {"tokens": len(control.split())}

        report[lang] = entry
        print(
            f"{lang}: naibbe ok; arithmetic round-trip="
            f"{'exact' if roundtrip_ok else 'MISMATCH'}, "
            f"doubling {stats['doubling_rate']:.4%} "
            f"(strength {tuned:.4f}, {'OK' if doubling_ok else 'OUT OF TOLERANCE'}); "
            f"control ok"
        )

    (out_dir / "acceptance_stats.json").write_text(json.dumps(report, indent=2))
    print(f"stats: {out_dir / 'acceptance_stats.json'}")
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    print("task 0.7 acceptance: all checks passed")


if __name__ == "__main__":
    main()
