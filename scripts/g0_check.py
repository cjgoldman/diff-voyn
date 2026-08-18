"""Gate G0 verification (end of Phase 0).

Checks, in order:

1. Vocab spec frozen: 32 symbols, spec hash printed, encode/decode round-trip.
2. Normalization stable on a round-trip across all languages: idempotent,
   whitespace-free, only frozen-alphabet chars, on real corpus samples.
3. Corpus table complete with size flags (from the corpus manifest).
4. Per-language held-out NELBO logs correctly to ClearML on a **random-init**
   model (the StubDenoiser — Phase 1 replaces it with the real backbone).
   Sanity anchor: random init must score near log2(32) = 5 bits/char.

Run: ``uv run python scripts/g0_check.py [--no-clearml]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from diff_voyn import normalize as norm
from diff_voyn import vocab
from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, CorpusWindows, LanguageSampler
from diff_voyn.infra.config import RunConfig
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char
from diff_voyn.model.stub import StubDenoiser

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    use_clearml = "--no-clearml" not in sys.argv
    root = data_root()
    corpus_dir = root / "corpora" / "v1"

    print("G0.1 vocab spec")
    check("32 symbols", vocab.VOCAB_SIZE == 32 and len(vocab.TOKENS) == 32)
    sample = "aequitas"
    check(
        "encode/decode round-trip",
        vocab.decode(vocab.encode(sample)) == sample,
    )
    print(f"  vocab spec sha256: {vocab.spec_hash()}")

    print("G0.2 normalization round-trip stability on real corpus text")
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    for lang in ("latin", "italian", "german"):
        doc = manifest["documents"][lang][0]
        text = (corpus_dir / lang / "docs" / f"{doc['doc_id']}.txt").read_text()[
            :200_000
        ]
        renorm = norm.normalize(text)
        check(f"{lang}: idempotent", renorm == text)
        check(f"{lang}: whitespace-free", not any(c.isspace() for c in text))
        check(f"{lang}: alphabet-closed", set(text) <= set(vocab.LETTERS))

    print("G0.3 corpus table")
    for lang, info in manifest["languages"].items():
        flag = " [LOW-RESOURCE]" if info["low_resource_flag"] else ""
        drop = info["norm_stats"]["letter_drop_rate"]
        check(
            f"{lang}: {info['norm_chars']:,} chars, drop {drop:.5%}{flag}",
            drop < 0.001,
        )

    print(
        "G0.4 per-language NELBO on random-init model"
        + (" -> ClearML" if use_clearml else " (local only)")
    )
    splits = load_splits(corpus_dir)
    heldout_ids = {
        lang: [d["doc_id"] for d in sp["heldout"]][:3]
        for lang, sp in splits["languages"].items()
    }
    windows = CorpusWindows(corpus_dir, heldout_ids)
    torch.manual_seed(0)
    model = StubDenoiser().eval()

    import numpy as np

    rng = np.random.default_rng(0)
    nelbo = {}
    for lang in LANG_TO_INDEX:
        ids = torch.from_numpy(
            np.stack([windows.sample_window(lang, 256, rng) for _ in range(4)])
        ).long()
        # seed=0 for every language: common random numbers (design §5a)
        nelbo[lang] = estimate_nelbo_bits_per_char(
            model, ids, LANG_TO_INDEX[lang], n_strata=16, seed=0
        )
        check(
            f"{lang}: random-init NELBO {nelbo[lang]:.3f} bits/char (~5 expected)",
            3.5 < nelbo[lang] < 7.0,
        )

    if use_clearml:
        from diff_voyn.infra.clearml_task import (
            init_task,
            report_language_weights,
            report_per_language_nelbo,
        )

        cfg = RunConfig(run_name="g0-check-random-init-nelbo", phase="phase0")
        task = init_task(cfg, root, tags=["g0"])
        sampler = LanguageSampler(
            {l: sp["train_chars"] for l, sp in splits["languages"].items()}
        )
        report_language_weights(task, sampler.weights_dict())
        report_per_language_nelbo(task, nelbo, iteration=0)
        task.get_logger().flush(wait=True)
        print(f"  ClearML task: {task.get_output_log_web_page()}")
        task.close()

    print()
    if FAILURES:
        print(f"G0: {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("G0: all checks passed")


if __name__ == "__main__":
    main()
