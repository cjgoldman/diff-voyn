"""R3 probe (task 5.2 / design §8): do the frozen evaluator's dense gradients
move a 1:1 key? On the short rung-1 cells where the n-gram search fails
(L = 50 / 100), start from the diffusion shortlist pick and compare
(a) soft expected-embedding refinement (stronger settings),
(b) straight-through refinement (hard forward, soft backward — the §8 fallback),
(c) ELBO-driven Gumbel–Sinkhorn search from scratch, soft and straight-through.
Each result is projected to a hard key and scored with paired masks.
Writes DATA_ROOT/analysis/phase5/r3_refinement_probe.{json,md}.
"""

from __future__ import annotations

import argparse
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
from diff_voyn.heads.ladder import (
    elbo_gradient_search,
    refine_assignment,
    write_json_atomic,
)
from diff_voyn.heads.two_tier import paired_bits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    p.add_argument("--lengths", type=int, nargs="+", default=[50, 100])
    p.add_argument("--languages", nargs="+", default=["latin", "italian"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument(
        "--ckpt", type=Path, default=root / "runs/phase_c-85m-seed0/ckpt_final.pt"
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(2)
    ev = DiffusionEvaluator.from_checkpoint(args.ckpt, device=args.device)
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
        "soft_refine_strong": {
            "kind": "refine",
            "straight_through": False,
            "steps": 50,
            "lr": 0.3,
            "n_strata": 8,
            "init_scale": 2.0,
        },
        "st_refine": {
            "kind": "refine",
            "straight_through": True,
            "steps": 50,
            "lr": 0.3,
            "n_strata": 8,
            "init_scale": 2.0,
        },
        "elbo_search_soft": {
            "kind": "search",
            "straight_through": False,
            "steps": 150,
            "n_strata": 4,
        },
        "elbo_search_st": {
            "kind": "search",
            "straight_through": True,
            "steps": 150,
            "n_strata": 4,
        },
    }
    rows = []
    t0 = time.time()
    for L in args.lengths:
        for lang in args.languages:
            inst = [r for r in scores if r["length"] == L and r["language"] == lang][
                : args.n
            ]
            for r in inst:
                s = solves[(lang, L, r["trial"])]
                cipher = np.asarray(s["cipher_ids"])
                plain = np.asarray(s["plain_ids"])
                h = r["hypotheses"][lang]
                start = (
                    np.asarray(h["diffusion"]["key"])
                    if "key" in h["diffusion"]
                    else None
                )
                # the pick's perm is not in as_dict; recover it from the solves shortlist by matching ser+ngram score
                if start is None:
                    for c in s["hypotheses"][lang]["shortlist"]:
                        if c["ngram_hard"] == h["diffusion"]["inner_score"]:
                            start = np.asarray(c["perm"])
                            break
                seed = zlib.crc32(f"r3/{lang}/{L}/{r['trial']}".encode()) % (2**31)
                rec = {
                    "language": lang,
                    "length": L,
                    "trial": r["trial"],
                    "ser_pick": float(np.mean(start[cipher] != plain)),
                    "bits_pick": h["diffusion"]["bits"][lang],
                    "ser_oracle": h["oracle"]["ser"],
                    "variants": {},
                }
                for name, v in variants.items():
                    t1 = time.time()
                    if v["kind"] == "refine":
                        perm, losses = refine_assignment(
                            ev,
                            cipher,
                            start,
                            language=lang,
                            bijective=True,
                            steps=v["steps"],
                            lr=v["lr"],
                            n_strata=v["n_strata"],
                            init_scale=v["init_scale"],
                            seed=seed,
                            straight_through=v["straight_through"],
                        )
                    else:
                        perm, losses = elbo_gradient_search(
                            ev,
                            cipher,
                            language=lang,
                            steps=v["steps"],
                            n_strata=v["n_strata"],
                            seed=seed,
                            straight_through=v["straight_through"],
                        )
                    bits = float(
                        paired_bits(
                            ev,
                            np.stack([perm[cipher], start[cipher]]),
                            [lang],
                            n_strata=64,
                            seed=seed,
                        )[:, 0][0]
                    )
                    bits_start = float(
                        paired_bits(
                            ev,
                            np.stack([perm[cipher], start[cipher]]),
                            [lang],
                            n_strata=64,
                            seed=seed,
                        )[:, 0][1]
                    )
                    rec["variants"][name] = {
                        "ser": float(np.mean(perm[cipher] != plain)),
                        "bits": bits,
                        "bits_start_same_masks": bits_start,
                        "changed": bool((perm != start).any()),
                        "n_changed": int((perm != start).sum()),
                        "loss_first_last": [losses[0], losses[-1]],
                        "seconds": round(time.time() - t1, 1),
                    }
                rows.append(rec)
                print(
                    f"{lang} L={L} t{r['trial']}: pick SER {rec['ser_pick']:.2f} | "
                    + " | ".join(
                        f"{n} SER {d['ser']:.2f} Δ{d['n_changed']} bits {d['bits']:.3f}"
                        for n, d in rec["variants"].items()
                    )
                    + f"  ({time.time()-t0:.0f}s)",
                    flush=True,
                )
    summary = {}
    for L in args.lengths:
        for lang in args.languages:
            rs = [x for x in rows if x["length"] == L and x["language"] == lang]
            summary[f"{lang}/L{L}"] = {
                "n": len(rs),
                "ser_pick": float(np.mean([x["ser_pick"] for x in rs])),
                "ser_oracle": float(np.mean([x["ser_oracle"] for x in rs])),
                "bits_pick": float(np.mean([x["bits_pick"] for x in rs])),
                **{
                    n: {
                        "ser": float(np.mean([x["variants"][n]["ser"] for x in rs])),
                        "bits": float(np.mean([x["variants"][n]["bits"] for x in rs])),
                        "changed_rate": float(
                            np.mean([x["variants"][n]["changed"] for x in rs])
                        ),
                        "better_ser_rate": float(
                            np.mean(
                                [x["variants"][n]["ser"] < x["ser_pick"] for x in rs]
                            )
                        ),
                        "worse_ser_rate": float(
                            np.mean(
                                [x["variants"][n]["ser"] > x["ser_pick"] for x in rs]
                            )
                        ),
                        "lower_bits_rate": float(
                            np.mean(
                                [
                                    x["variants"][n]["bits"]
                                    < x["variants"][n]["bits_start_same_masks"]
                                    for x in rs
                                ]
                            )
                        ),
                    }
                    for n in variants
                },
            }
    write_json_atomic(
        root / "analysis/phase5/r3_refinement_probe.json",
        {"variants": variants, "summary": summary, "instances": rows},
    )
    md = [
        "### R3 probe — dense ELBO gradients on 1:1 keys (short cells)",
        "",
        "| cell | n | SER pick (oracle) | "
        + " | ".join(
            f"{n}: SER / changed / better / worse / lower-bits" for n in variants
        )
        + " |",
        "|---|---|---|" + "---|" * len(variants),
    ]
    for k, c in summary.items():
        md.append(
            f"| {k} | {c['n']} | {c['ser_pick']:.3f} ({c['ser_oracle']:.3f}) | "
            + " | ".join(
                f"{c[n]['ser']:.3f} / {c[n]['changed_rate']:.0%} / {c[n]['better_ser_rate']:.0%} / {c[n]['worse_ser_rate']:.0%} / {c[n]['lower_bits_rate']:.0%}"
                for n in variants
            )
            + " |"
        )
    (root / "analysis/phase5/r3_refinement_probe.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
