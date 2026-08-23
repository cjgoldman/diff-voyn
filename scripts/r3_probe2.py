"""R3 probe, part 2: conditioned refinement (fixed masks, near-hard init,
small lr, best-of-trajectory) — soft vs straight-through — on the same short
rung-1 cells. Writes analysis/phase5/r3_probe2.{json,md}."""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
from diff_voyn.heads.ladder import refine_assignment_tracked, write_json_atomic

root = data_root()
torch.set_float32_matmul_precision("high")
torch.set_num_threads(2)
ev = DiffusionEvaluator.from_checkpoint(
    root / "runs/phase_c-85m-seed0/ckpt_final.pt", device="cuda"
)
scores = json.loads((root / "analysis/phase5/rung1_scores.json").read_text())[
    "instances"
]
solves = {
    (r["language"], r["length"], r["trial"]): r
    for r in json.loads((root / "analysis/phase5/rung1_solves.json").read_text())[
        "instances"
    ]
}
variants = {
    "fixed_soft": {"straight_through": False, "fixed_masks": True},
    "fixed_st": {"straight_through": True, "fixed_masks": True},
    "fresh_soft_lowlr": {"straight_through": False, "fixed_masks": False},
}
rows = []
t0 = time.time()
for L in (50, 100):
    for lang in ("latin", "italian"):
        for r in [x for x in scores if x["length"] == L and x["language"] == lang][:8]:
            s = solves[(lang, L, r["trial"])]
            cipher = np.asarray(s["cipher_ids"])
            plain = np.asarray(s["plain_ids"])
            h = r["hypotheses"][lang]
            start = next(
                np.asarray(c["perm"])
                for c in s["hypotheses"][lang]["shortlist"]
                if c["ngram_hard"] == h["diffusion"]["inner_score"]
            )
            seed = zlib.crc32(f"r3b/{lang}/{L}/{r['trial']}".encode()) % (2**31)
            rec = {
                "language": lang,
                "length": L,
                "trial": r["trial"],
                "ser_pick": float(np.mean(start[cipher] != plain)),
                "ser_oracle": h["oracle"]["ser"],
                "variants": {},
            }
            for name, v in variants.items():
                key, info = refine_assignment_tracked(
                    ev, cipher, start, language=lang, bijective=True, seed=seed, **v
                )
                rec["variants"][name] = {
                    "ser": float(np.mean(key[cipher] != plain)),
                    "n_changed": int((key != start).sum()),
                    "bits_best": info["best_bits"],
                    "bits_start": info["trace"][0]["bits_start"],
                    "trace": info["trace"],
                }
            rows.append(rec)
            print(
                f"{lang} L={L} t{r['trial']}: pick SER {rec['ser_pick']:.2f} | "
                + " | ".join(
                    f"{n} SER {d['ser']:.2f} Δ{d['n_changed']} bits {d['bits_start']:.3f}->{d['bits_best']:.3f}"
                    for n, d in rec["variants"].items()
                )
                + f" ({time.time()-t0:.0f}s)",
                flush=True,
            )
summary = {}
for L in (50, 100):
    for lang in ("latin", "italian"):
        rs = [x for x in rows if x["length"] == L and x["language"] == lang]
        summary[f"{lang}/L{L}"] = {
            "n": len(rs),
            "ser_pick": float(np.mean([x["ser_pick"] for x in rs])),
            "ser_oracle": float(np.mean([x["ser_oracle"] for x in rs])),
            **{
                n: {
                    "ser": float(np.mean([x["variants"][n]["ser"] for x in rs])),
                    "changed_rate": float(
                        np.mean([x["variants"][n]["n_changed"] > 0 for x in rs])
                    ),
                    "better_ser_rate": float(
                        np.mean([x["variants"][n]["ser"] < x["ser_pick"] for x in rs])
                    ),
                    "worse_ser_rate": float(
                        np.mean([x["variants"][n]["ser"] > x["ser_pick"] for x in rs])
                    ),
                    "bits_gain": float(
                        np.mean(
                            [
                                x["variants"][n]["bits_start"]
                                - x["variants"][n]["bits_best"]
                                for x in rs
                            ]
                        )
                    ),
                }
                for n in variants
            },
        }
write_json_atomic(
    root / "analysis/phase5/r3_probe2.json",
    {"variants": variants, "summary": summary, "instances": rows},
)
md = [
    "### R3 probe 2 — conditioned refinement (fixed masks, init 6, lr 0.05, 40 steps, best-of-trajectory)",
    "",
    "| cell | n | SER pick (oracle) | "
    + " | ".join(f"{n}: SER / changed / better / worse / bits gain" for n in variants)
    + " |",
    "|---|---|---|" + "---|" * len(variants),
]
for k, c in summary.items():
    md.append(
        f"| {k} | {c['n']} | {c['ser_pick']:.3f} ({c['ser_oracle']:.3f}) | "
        + " | ".join(
            f"{c[n]['ser']:.3f} / {c[n]['changed_rate']:.0%} / {c[n]['better_ser_rate']:.0%} / {c[n]['worse_ser_rate']:.0%} / {c[n]['bits_gain']:.3f}"
            for n in variants
        )
        + " |"
    )
(root / "analysis/phase5/r3_probe2.md").write_text("\n".join(md))
print("\n".join(md))
