"""Score a backbone checkpoint on the held-out splits — the no-manual-steps
harness required by task 1.3 ("harness scores pilot checkpoints without
manual intervention"); grows into the full Phase-3 scoring harness (3.1).

Per language: conditional held-out NELBO (bits/char) and the unconditional
(NULL-language) NELBO, with common random numbers across every condition —
the same fixed windows and masking seed the training canary uses, so numbers
are directly comparable to the ClearML curves. Uses the EMA weights by
default (design §7.5: EMA is what every evaluation uses).

Usage:
    uv run python scripts/score_checkpoint.py DATA_ROOT/runs/<run>/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.splits import load_splits
from diff_voyn.data.loader import LANG_TO_INDEX, NULL_LANG_INDEX, CorpusWindows
from diff_voyn.infra.config import ModelConfig
from diff_voyn.infra.nelbo import estimate_nelbo_bits_per_char
from diff_voyn.model.backbone import Backbone

EVAL_WINDOW_SEED = 12345  # keep in sync with scripts/train.py
EVAL_NELBO_SEED = 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--windows", type=int, default=8, help="per language")
    p.add_argument("--strata", type=int, default=64)
    p.add_argument("--seed", type=int, default=EVAL_NELBO_SEED)
    p.add_argument("--raw", action="store_true", help="score raw (non-EMA) weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**state["extra"]["config"]["model"])
    model = Backbone(model_cfg).to(args.device).eval()
    model.load_state_dict(state["model"])
    source = "raw"
    if not args.raw and state.get("ema") is not None:
        sd = model.state_dict()
        for k, v in state["ema"]["shadow"].items():
            sd[k].copy_(v.to(sd[k].dtype))
        source = "ema"
    print(
        f"{args.checkpoint} @ step {state['step']} ({source} weights, "
        f"{model.n_params()/1e6:.1f}M params)"
    )

    root = data_root()
    corpus_dir = root / "corpora" / state["extra"]["config"]["data"]["corpus_version"]
    splits = load_splits(corpus_dir, state["extra"]["config"]["data"]["splits_version"])
    heldout_ids = {
        lang: [d["doc_id"] for d in sp["heldout"]]
        for lang, sp in splits["languages"].items()
    }
    windows = CorpusWindows(corpus_dir, heldout_ids)
    rng = np.random.default_rng(EVAL_WINDOW_SEED)
    batches = {
        lang: torch.from_numpy(
            np.stack(
                [
                    windows.sample_window(lang, model_cfg.seq_len, rng)
                    for _ in range(args.windows)
                ]
            ).astype(np.int64)
        )
        for lang in LANG_TO_INDEX
    }

    print(f"{'language':10s} {'cond':>8s} {'uncond':>8s} {'Δ(u−c)':>8s}   bits/char")
    for lang, ids in batches.items():
        common = {"n_strata": args.strata, "seed": args.seed, "device": args.device}
        cond = estimate_nelbo_bits_per_char(model, ids, LANG_TO_INDEX[lang], **common)
        uncond = estimate_nelbo_bits_per_char(model, ids, NULL_LANG_INDEX, **common)
        print(f"{lang:10s} {cond:8.4f} {uncond:8.4f} {uncond - cond:+8.4f}")


if __name__ == "__main__":
    main()
