"""Run the cipher-head synthetic grid — tasks CH.2 / CH.3 / CH.5 / CH.9.

Solves synthetic ciphers (1:1 substitution, unigram homophonic) built from
HELD-OUT corpus windows, under the frozen n-gram evaluator, and reports SER /
map accuracy / cost per (kind x language x length) cell; optionally the
trial-decipherment language-ranking probe. Results land under
``DATA_ROOT/cipher_heads/``.

Examples:
  uv run python scripts/cipher_head_harness.py --kinds sub1to1 --trials 5
  uv run python scripts/cipher_head_harness.py --kinds homophonic \\
      --lengths 408 --trials 3 --probe-language
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.heads.evaluator import NgramEvaluator
from diff_voyn.heads.harness import (
    DEFAULT_LENGTHS,
    ngram_calibration_offsets,
    run_grid,
    save_results,
    summarize,
)
from diff_voyn.heads.ngram import lm_dir, load_lm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["sub1to1", "homophonic"])
    ap.add_argument("--languages", nargs="+", default=sorted(LANG_TO_INDEX))
    ap.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--n-symbols", type=int, default=54)
    ap.add_argument("--probe-language", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # the Phase-A training runs own the cores; small-tensor work is fastest
    # single-threaded (measured 24x vs default thread count)
    torch.set_num_threads(1)

    corpus_dir = data_root() / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    lms = {lang: load_lm(lm_dir() / f"{lang}.npz") for lang in args.languages}
    ev = NgramEvaluator(lms, calibration_offsets_bits=ngram_calibration_offsets(lms))

    results = run_grid(
        ev,
        corpus_dir,
        splits,
        kinds=tuple(args.kinds),
        languages=tuple(args.languages),
        lengths=tuple(args.lengths),
        trials=args.trials,
        n_symbols=args.n_symbols,
        probe_language=args.probe_language,
        seed=args.seed,
    )
    out = args.out or (
        data_root()
        / "cipher_heads"
        / f"grid_{'-'.join(args.kinds)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_results(results, out)
    print(json.dumps(summarize(results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
