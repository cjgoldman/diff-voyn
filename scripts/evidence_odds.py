"""Evidence odds for the manuscript cells — recommendation R1 of the Bayesian
review (docs/bayesian_perspective_review.md §3, 2026-09-02).

Turns the recorded control batteries into *reference piles* of the judge's
structure margin and reports, for every manuscript cell, how much more likely
its margin is under "a real cipher of this kind, through this solver" than
under "not a cipher of this kind" (a likelihood ratio; a Bayes factor once a
prior is attached). Nothing is re-scored: every number is read from the
recorded judge artifacts.

Two tables, one per solver family, because the piles must come from the
same key search as the manuscript number:

  wordhom   the solver of record (hapax-wildcard → anneal, post-all loop) on
            the word-homophonic head. Piles per hypothesis language from the
            manuscript-shaped battery (docs/alt_loop_plan.md §10; judge rows
            in DATA_ROOT/analysis/altloop/judge_at_ser_battery_*.json) and the
            A-like anneal finals (judge_at_ser.json). Variants of "real":
            clean (positive/nodouble/revdouble), mixed (80 % host + 20 %
            foreign block), dirty5, dirty10. "Not a cipher": shuffled,
            voynichesque, and the true cipher under the *wrong* language.
            Manuscript: DATA_ROOT/analysis/altloop_vms/runs_wordhom_anneal.json,
            arm ``post`` (the battery's ``post-all``), both seeds.
            The d5b20 decoder runs (``judge_at_ser_battery_big*``) are a
            different hypothesis space and are excluded.

  phase6    the Phase-6 heads (sub1to1 / homophonic / naibbe, ELBO-polished
            keys). Piles per (head, hypothesis language) from
            analysis/phase6/controls/report.json (positives; shuffled;
            voynichesque = real-text source, a wrong-hypothesis control;
            contamination = untrained language) and
            analysis/phase6/controls_nocontent/report.json (strict twins).
            Manuscript: analysis/phase6/vms_report.json (87 cells).

Likelihood model: each pile is summarised by a Student-t predictive (mean,
sd with a floor of ``--sd-floor`` bits to reflect the judge's own replicate
noise and single-seed piles, df = n − 1, scale inflated by sqrt(1 + 1/n)).
Piles are small (3–12 cells), so the ratios are order-of-magnitude
statements; the nonparametric position of the manuscript inside each pile
(fraction of pile cells at or below it) is printed beside them. The margin
is the only load-bearing half of ``ABSTAIN_RULE`` (docs/project_status.md
§4), so it is the statistic used; plain bits are reported but not modelled.

Power line: the fraction of each "real" pile that the frozen judge CALLED
(``called`` in the judge rows; ``language_like`` and the top language equal
to the truth for the Phase-6 cells).

Usage:
    uv run python scripts/evidence_odds.py            # both tables
    uv run python scripts/evidence_odds.py --family wordhom --sd-floor 0.15

Artifacts: DATA_ROOT/analysis/evidence_odds/odds.json, odds.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import data_root


def _load(path) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


LANGS = ("german", "latin", "italian")
NEG_LABELS = (
    "no_content",
    "wrong_language",
    "strict_twin",
    "shuffled",
    "contamination",
    "voynichesque_realtext",
)
REAL_LABELS = ("clean", "mixed", "dirty5", "dirty10")


# ---------------------------------------------------------------- likelihood


def _t_logpdf(x: float, df: float, loc: float, scale: float) -> float:
    z = (x - loc) / scale
    return (
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - (df + 1) / 2 * math.log1p(z * z / df)
    )


class Pile:
    """A reference pile of structure margins with a t-predictive density."""

    def __init__(self, label: str, rows: list[dict], sd_floor: float):
        self.label = label
        self.rows = rows
        self.x = [r["structure_margin"] for r in rows]
        self.n = len(self.x)
        self.mean = sum(self.x) / self.n if self.n else float("nan")
        if self.n >= 2:
            var = sum((v - self.mean) ** 2 for v in self.x) / (self.n - 1)
            self.sd = max(math.sqrt(var), sd_floor)
        else:
            self.sd = max(2.5 * sd_floor, sd_floor)
        self.called = (
            sum(1 for r in rows if r.get("called")) / self.n if self.n else float("nan")
        )

    def logpdf(self, x: float) -> float:
        if self.n == 0:
            return float("nan")
        df = max(self.n - 1, 1)
        scale = self.sd * math.sqrt(1.0 + 1.0 / self.n)
        return _t_logpdf(x, df, self.mean, scale)

    def frac_at_or_below(self, x: float) -> float:
        return sum(1 for v in self.x if v <= x) / self.n if self.n else float("nan")

    def summary(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "min": min(self.x) if self.x else None,
            "max": max(self.x) if self.x else None,
            "called_fraction": self.called,
            "cells": [
                {
                    k: r.get(k)
                    for k in (
                        "cell",
                        "key",
                        "structure_margin",
                        "plain_bits",
                        "ser",
                        "called",
                    )
                }
                for r in self.rows
            ],
        }


def odds_row(x: float, real: dict[str, Pile], neg: Pile) -> dict:
    out = {
        "margin": x,
        "neg_pile_n": neg.n,
        "frac_neg_at_or_below": neg.frac_at_or_below(x),
    }
    lp_neg = neg.logpdf(x)
    for lab, p in real.items():
        if p.n == 0:
            continue
        lr = (p.logpdf(x) - lp_neg) / math.log(10.0)
        out[lab] = {
            "n": p.n,
            "log10_LR_real_vs_not": lr,
            "frac_real_at_or_below": p.frac_at_or_below(x),
            "power_called": p.called,
        }
    return out


# ------------------------------------------------------------ wordhom family


def load_wordhom_battery(root: Path) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for fn in sorted(
        glob.glob(str(root / "analysis/altloop/judge_at_ser_battery_*.json"))
    ):
        name = Path(fn).name.replace("judge_at_ser_battery_", "")
        if name.startswith("big"):  # d5b20 decoder — a different hypothesis space
            continue
        for r in _load(fn):
            if not r["key"].startswith("final:_bat_anneal"):
                continue
            k = (r["cell"], r["hypothesis"], r["key"])
            if k in seen:
                continue
            seen.add(k)
            r = dict(r)
            r["source"] = Path(fn).name
            rows.append(r)
    # A-like clean positives: the §8.6 anneal finals scored in judge_at_ser.json
    for r in _load(root / "analysis/altloop/judge_at_ser.json"):
        if r["key"].startswith("wild:anneal"):
            r = dict(r)
            r["hypothesis"] = r["truth_language"]
            r["control"] = "positive"
            r["source"] = "judge_at_ser.json"
            rows.append(r)
    return rows


def wordhom_label(r: dict) -> str:
    if r["hypothesis"] != r["truth_language"]:
        return "wrong_language"
    c = r["control"]
    if c in ("shuffled", "voynichesque"):
        return "no_content"
    if c == "dirty":
        return "dirty5" if "_s05" in r["cell"] else "dirty10"
    if c == "mixed":
        return "mixed"
    if c in ("positive", "nodouble", "revdouble"):
        return "clean"
    raise ValueError(f"unlabelled battery row {r['cell']} {c}")


def wordhom_table(root: Path, sd_floor: float) -> dict:
    rows = load_wordhom_battery(root)
    by_lang: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_lang[r["hypothesis"]][wordhom_label(r)].append(r)

    piles: dict[str, dict] = {}
    real: dict[str, dict[str, Pile]] = {}
    neg: dict[str, Pile] = {}
    for lang in LANGS:
        groups = by_lang[lang]
        neg[lang] = Pile(
            "not_a_cipher", groups["no_content"] + groups["wrong_language"], sd_floor
        )
        real[lang] = {lab: Pile(lab, groups[lab], sd_floor) for lab in REAL_LABELS}
        piles[lang] = {
            "not_a_cipher": neg[lang].summary(),
            "no_content_only": Pile(
                "no_content", groups["no_content"], sd_floor
            ).summary(),
            "wrong_language_only": Pile(
                "wrong_language", groups["wrong_language"], sd_floor
            ).summary(),
            **{lab: real[lang][lab].summary() for lab in REAL_LABELS},
        }

    runs = _load(root / "analysis/altloop_vms/runs_wordhom_anneal.json")["runs"]
    cells = []
    for run in runs:
        _, rest = run["cell"].split(":", 1)
        transcription, dialect, _w, lang = rest.split("/")
        fm = run["final_metrics"]
        x = fm["structure_margin"]
        entry = {
            "cell": run["cell"],
            "transcription": transcription,
            "dialect": dialect,
            "hypothesis": lang,
            "arm": run["arm"],
            "seed": run["seed"],
            "plain_bits": fm["plain_bits"],
            "language_like": fm["language_like"],
            "odds": odds_row(x, real[lang], neg[lang]),
        }
        cells.append(entry)
    return {"piles": piles, "cells": cells}


# ------------------------------------------------------------- phase6 family


def phase6_called(c: dict, truth: str | None) -> bool:
    return bool(c["language_like"]) and (
        truth is None or c["top_language_of_decode"] == truth
    )


def phase6_table(root: Path, sd_floor: float) -> dict:
    ctl = _load(root / "analysis/phase6/controls/report.json")
    nc = _load(root / "analysis/phase6/controls_nocontent/report.json")
    vms = _load(root / "analysis/phase6/vms_report.json")
    truth_of = {v["instance"]: v["truth_language"] for v in ctl["verdicts"]}

    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in ctl["cells"]:
        head, hyp, inst = c["head"], c["hypothesis"], c["instance"]
        truth = truth_of.get(inst)
        r = {
            "cell": f"{inst}/{c['presentation']}",
            "key": c.get("final_source"),
            "structure_margin": c["structure_margin"],
            "plain_bits": c["plain_bits"],
            "ser": None,
            "called": phase6_called(c, truth),
        }
        if c["control"] == "positive":
            groups[(head, hyp)]["clean" if truth == hyp else "wrong_language"].append(r)
        elif c["control"] == "shuffled":
            groups[(head, hyp)]["shuffled"].append(r)
        elif c["control"] == "voynichesque":
            groups[(head, hyp)]["voynichesque_realtext"].append(r)
        elif c["control"] == "contamination":
            groups[(head, hyp)]["contamination"].append(r)
    for c in nc["cells"]:
        r = {
            "cell": f"{c['instance']}/{c['presentation']}",
            "key": c.get("final_source"),
            "structure_margin": c["structure_margin"],
            "plain_bits": c["plain_bits"],
            "ser": None,
            "called": phase6_called(c, None),
        }
        groups[(c["head"], c["hypothesis"])]["strict_twin"].append(r)

    piles: dict[str, dict] = {}
    real: dict[tuple[str, str], dict[str, Pile]] = {}
    neg: dict[tuple[str, str], Pile] = {}
    for key, g in groups.items():
        # strict negatives: letter-shuffled-source twins + shuffled text + the
        # true cipher under the wrong language; voynichesque (real-text source)
        # and contamination are reported as their own piles, not pooled.
        neg[key] = Pile(
            "not_a_cipher",
            g["strict_twin"] + g["shuffled"] + g["wrong_language"],
            sd_floor,
        )
        real[key] = {"clean": Pile("clean", g["clean"], sd_floor)}
        piles["/".join(key)] = {
            "not_a_cipher": neg[key].summary(),
            "strict_twin_only": Pile(
                "strict_twin", g["strict_twin"], sd_floor
            ).summary(),
            "voynichesque_realtext": Pile(
                "voynichesque_realtext", g["voynichesque_realtext"], sd_floor
            ).summary(),
            "contamination": Pile(
                "contamination", g["contamination"], sd_floor
            ).summary(),
            "clean": real[key]["clean"].summary(),
        }

    cells = []
    for c in vms["cells"]:
        key = (c["head"], c["hypothesis"])
        if key not in neg or neg[key].n == 0 or real[key]["clean"].n == 0:
            continue  # arithmetic head has no control piles
        x = c["structure_margin"]
        cells.append(
            {
                "cell": f"{c['head']}/{c['instance']}/{c['presentation']}/{c['hypothesis']}",
                "head": c["head"],
                "instance": c["instance"],
                "presentation": c["presentation"],
                "hypothesis": c["hypothesis"],
                "plain_bits": c["plain_bits"],
                "language_like": c["language_like"],
                "odds": odds_row(x, real[key], neg[key]),
                "voynichesque_realtext_frac_at_or_below": (
                    Pile(
                        "v", groups[key]["voynichesque_realtext"], sd_floor
                    ).frac_at_or_below(x)
                    if groups[key]["voynichesque_realtext"]
                    else None
                ),
            }
        )
    return {"piles": piles, "cells": cells}


# ------------------------------------------------------------------- report


def _f(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _lr(v):
    if v is None:
        return "—"
    if v > 3:
        return "> 1000"
    if v < -3:
        return "< 1/1000"
    return f"{10 ** v:.2g}" if v >= 0 else f"1/{10 ** -v:.2g}"


def md_wordhom(t: dict) -> list[str]:
    out = [
        "## Word-homophonic head, solver of record (wild → anneal, post-all, arm `post`)",
        "",
    ]
    out.append(
        "Reference piles (structure margin of the anneal final, bits/char). `called` = fraction the frozen judge called."
    )
    out.append("")
    out.append("| hypothesis | pile | n | mean ± sd | range | called |")
    out.append("|---|---|---|---|---|---|")
    for lang in LANGS:
        for lab in (
            "not_a_cipher",
            "no_content_only",
            "wrong_language_only",
            "clean",
            "mixed",
            "dirty5",
            "dirty10",
        ):
            p = t["piles"][lang][lab]
            if p["n"] == 0:
                continue
            out.append(
                f"| {lang} | {lab} | {p['n']} | {_f(p['mean'])} ± {_f(p['sd'])} | "
                f"{_f(p['min'])}–{_f(p['max'])} | {_f(p['called_fraction'])} |"
            )
    out.append("")
    out.append(
        "Manuscript cells (arm `post`, both seeds). LR = likelihood of the cell's margin under the real pile ÷ under the not-a-cipher pile. `pos` = fraction of that pile at or below the manuscript's margin."
    )
    out.append("")
    out.append(
        "| cell | seed | margin | pos in not-a-cipher | LR clean (pos) | LR mixed (pos) | LR dirty5 (pos) | LR dirty10 (pos) | power clean / dirty5 / dirty10 |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for c in sorted(
        t["cells"],
        key=lambda c: (c["dialect"], c["transcription"], c["hypothesis"], c["seed"]),
    ):
        if c["arm"] != "post":
            continue
        o = c["odds"]

        def col(lab, o=o):
            if lab not in o:
                return "—"
            return f"{_lr(o[lab]['log10_LR_real_vs_not'])} ({_f(o[lab]['frac_real_at_or_below'])})"

        pw = " / ".join(
            _f(o[l]["power_called"]) if l in o else "—"
            for l in ("clean", "dirty5", "dirty10")
        )
        name = f"{c['transcription']}/{c['dialect']} :{c['hypothesis']}"
        out.append(
            f"| {name} | {c['seed']} | {_f(o['margin'])} | {_f(o['frac_neg_at_or_below'])} | "
            f"{col('clean')} | {col('mixed')} | {col('dirty5')} | {col('dirty10')} | {pw} |"
        )
    # sensitivity across arms
    lo = min(c["odds"]["margin"] for c in t["cells"])
    hi = max(c["odds"]["margin"] for c in t["cells"])
    out.append("")
    out.append(
        f"Across all four arms × two seeds the manuscript's anneal-final margins span {_f(lo)}–{_f(hi)}."
    )
    return out


def md_phase6(t: dict) -> list[str]:
    out = ["## Phase-6 heads (ELBO-polished keys; sub1to1 / homophonic / naibbe)", ""]
    out.append(
        "Reference piles per (head, hypothesis). `not_a_cipher` = strict twins + shuffled + true cipher under the wrong language. Voynichesque (real-text source) and contamination are shown separately."
    )
    out.append("")
    out.append(
        "| head/hyp | not-a-cipher n, mean ± sd, max | strict twins max | voynichesque real-text range | contamination range | clean n, mean, min | clean called |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for key in sorted(t["piles"]):
        p = t["piles"][key]
        na, st, vr, co, cl = (
            p[k]
            for k in (
                "not_a_cipher",
                "strict_twin_only",
                "voynichesque_realtext",
                "contamination",
                "clean",
            )
        )
        out.append(
            f"| {key} | {na['n']}, {_f(na['mean'])} ± {_f(na['sd'])}, {_f(na['max'])} | {_f(st['max'])} | "
            f"{_f(vr['min'])}–{_f(vr['max'])} | {_f(co['min'])}–{_f(co['max'])} | "
            f"{cl['n']}, {_f(cl['mean'])}, {_f(cl['min'])} | {_f(cl['called_fraction'])} |"
        )
    out.append("")
    out.append(
        "Manuscript cells. LR = clean-real ÷ not-a-cipher. `pos neg` / `pos clean` / `pos voyn` = fraction of that pile at or below the cell's margin."
    )
    out.append("")
    out.append(
        "| cell | margin | plain bits | pos neg | LR clean | pos clean | pos voyn (real-text) |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for c in sorted(
        t["cells"],
        key=lambda c: (c["instance"], c["head"], c["presentation"], c["hypothesis"]),
    ):
        o = c["odds"]
        cl = o.get("clean", {})
        out.append(
            f"| {c['cell']} | {_f(o['margin'])} | {_f(c['plain_bits'])} | {_f(o['frac_neg_at_or_below'])} | "
            f"{_lr(cl.get('log10_LR_real_vs_not'))} | {_f(cl.get('frac_real_at_or_below'))} | "
            f"{_f(c['voynichesque_realtext_frac_at_or_below'])} |"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--family", choices=("wordhom", "phase6", "both"), default="both")
    ap.add_argument(
        "--sd-floor",
        type=float,
        default=0.10,
        help="floor on a pile's sd, bits/char (judge replicate noise ≈ 0.07)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = data_root()
    out_dir = args.out or (root / "analysis/evidence_odds")
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "sd_floor": args.sd_floor,
        "statistic": "structure_margin (bits/char)",
        "likelihood": "Student-t predictive per pile",
    }
    md = [
        "# Evidence odds for the manuscript cells (R1 of docs/bayesian_perspective_review.md)",
        "",
    ]
    md.append(
        f"Statistic: structure margin. Likelihood per pile: Student-t predictive (df = n − 1, sd floor {args.sd_floor}). Piles are small; read ratios as orders of magnitude."
    )
    md.append("")
    if args.family in ("wordhom", "both"):
        result["wordhom"] = wordhom_table(root, args.sd_floor)
        md += md_wordhom(result["wordhom"]) + [""]
    if args.family in ("phase6", "both"):
        result["phase6"] = phase6_table(root, args.sd_floor)
        md += md_phase6(result["phase6"]) + [""]

    (out_dir / "odds.json").write_text(json.dumps(result, indent=1))
    (out_dir / "odds.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out_dir / 'odds.json'} and odds.md")


if __name__ == "__main__":
    main()
