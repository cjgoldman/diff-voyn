"""Train + persist the per-language char n-gram LMs — task CH.0.

Trains interpolated Witten-Bell models (orders 1..5) on the TRAIN side of
splits v1 for each frozen language, reports held-out bits/char (the
acceptance number), and persists tables under DATA_ROOT/ngram_lms/v1/.

Usage: uv run python scripts/train_ngram_lms.py [--k-max 5]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.heads.ngram import lm_dir, save_lm, train_from_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=5)
    args = ap.parse_args()

    corpus_dir = data_root() / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    out_dir = lm_dir()
    summary = {}
    for lang in LANG_TO_INDEX:
        t0 = time.time()
        lm, bits = train_from_corpus(corpus_dir, splits, lang, k_max=args.k_max)
        path = save_lm(lm, out_dir)
        per_order = {
            k: round(
                sum(
                    -lm.score_ids(_read(corpus_dir, lang, d["doc_id"]), k)
                    for d in splits["languages"][lang]["heldout"]
                )
                / (splits["languages"][lang]["heldout_chars"] * 0.6931471805599453),
                4,
            )
            for k in range(1, args.k_max + 1)
        }
        summary[lang] = {
            "heldout_bits_per_char": round(bits, 4),
            "per_order_bits": per_order,
            "path": str(path),
            "train_seconds": round(time.time() - t0, 1),
        }
        print(f"{lang}: {bits:.4f} bits/char held-out  per-order={per_order}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/summary.json")


def _read(corpus_dir: Path, lang: str, doc_id: str):
    from diff_voyn.heads.ngram import encode_letters

    return encode_letters((corpus_dir / lang / "docs" / f"{doc_id}.txt").read_text())


if __name__ == "__main__":
    main()
