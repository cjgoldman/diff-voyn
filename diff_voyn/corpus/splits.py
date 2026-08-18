"""Held-out calibration splits — task 0.4 (design §5b).

Carved once, from the corpus manifest, *before any training touches the data*;
versioned and content-addressed so no later step can silently move a document
across the boundary.

Policy:

- Document-level: a work is entirely train or entirely held-out (no window of
  a held-out work ever appears in training).
- Domain-matched: within each language, documents are selected per source
  domain proportionally to that domain's share of the language's characters,
  so the held-out set mirrors the language's own domain mix.
- Size-matched: the same character target for every language (default 500k,
  well above the ≥200k acceptance floor), so cross-language calibration (§5b)
  compares equal-sized samples.
- Deterministic: seeded RNG over sorted doc ids; the split file records the
  seed, the exact doc-id lists, and each document's sha256.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLITS_VERSION = "v1"
HELDOUT_TARGET_CHARS = 500_000
HELDOUT_MIN_CHARS = 200_000  # acceptance floor from task 0.4


def carve_splits(
    manifest_path: Path,
    seed: int = 20260818,
    target_chars: int = HELDOUT_TARGET_CHARS,
) -> Path:
    manifest = json.loads(manifest_path.read_text())
    rng = random.Random(seed)
    splits: dict[str, dict] = {}

    for lang, docs in manifest["documents"].items():
        docs = sorted(docs, key=lambda d: d["doc_id"])
        total = sum(d["norm_chars"] for d in docs)
        by_domain: dict[str, list[dict]] = {}
        for d in docs:
            by_domain.setdefault(d["domain"], []).append(d)

        heldout: list[dict] = []
        got = 0
        # Per-domain character quota proportional to domain share. A domain
        # whose every document exceeds the size cap is skipped rather than
        # force-picked — holding out a giant canonical work (e.g. the Luther
        # Bible, 22% of Italian's Decameron) would waste prime training text;
        # the global top-up below covers any shortfall from other domains.
        for domain, ddocs in sorted(by_domain.items()):
            domain_chars = sum(d["norm_chars"] for d in ddocs)
            quota = target_chars * domain_chars / total
            cap = max(quota * 1.5, 60_000)
            candidates = [d for d in ddocs if d["norm_chars"] <= cap]
            rng.shuffle(candidates)
            picked_chars = 0
            for d in candidates:
                if picked_chars >= quota:
                    break
                heldout.append(d)
                picked_chars += d["norm_chars"]
            got += picked_chars

        # Top up from any domain if under the acceptance floor.
        if got < HELDOUT_MIN_CHARS:
            chosen = {d["doc_id"] for d in heldout}
            rest = sorted(
                (d for d in docs if d["doc_id"] not in chosen),
                key=lambda d: d["norm_chars"],
            )
            for d in rest:
                if got >= HELDOUT_MIN_CHARS:
                    break
                heldout.append(d)
                got += d["norm_chars"]

        heldout_ids = {d["doc_id"] for d in heldout}
        train = [d for d in docs if d["doc_id"] not in heldout_ids]
        assert not heldout_ids & {d["doc_id"] for d in train}
        splits[lang] = {
            "heldout_chars": sum(d["norm_chars"] for d in heldout),
            "train_chars": sum(d["norm_chars"] for d in train),
            "heldout": [
                {"doc_id": d["doc_id"], "sha256": d["sha256"], "chars": d["norm_chars"]}
                for d in sorted(heldout, key=lambda d: d["doc_id"])
            ],
            "train": [
                {"doc_id": d["doc_id"], "sha256": d["sha256"], "chars": d["norm_chars"]}
                for d in train
            ],
        }

    out = {
        "splits_version": SPLITS_VERSION,
        "corpus_version": manifest["corpus_version"],
        "seed": seed,
        "target_chars": target_chars,
        "languages": splits,
    }
    path = manifest_path.parent / f"splits_{SPLITS_VERSION}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def load_splits(corpus_dir: Path, version: str = SPLITS_VERSION) -> dict:
    return json.loads((corpus_dir / f"splits_{version}.json").read_text())
