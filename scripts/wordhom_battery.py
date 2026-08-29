"""Manuscript-shaped word-homophonic battery for the wildcard→anneal
pipeline (docs/alt_loop_plan.md §8.4–8.7): the controls the A-like
positives were validated without.

Instances (all ``wordtypesall``, one per language and shape; ``Alike`` =
14 000 letters / 5 200 key types (≈ 4.1 tokens per type, Currier A),
``Blike`` = 30 000 / 7 200 (≈ 5.5, Currier B)):

  shuffled/<lang>/{Alike,Blike}      letters permuted, then enciphered — a
                                     NEGATIVE with the language's unigram
                                     statistics and the manuscript's shape
  voynichesque/<lang>/{Alike,Blike}  pinned voynichesque gibberish at the
                                     manuscript's token count — NEGATIVE
  dirty/<lang>/Alike_s05, _s10       plaintext with i.i.d. transcription
                                     errors (Phase-2 ``TranscriptionNoise``,
                                     5 % / 10 % per character) enciphered
                                     under the clean key — POSITIVE whose
                                     truth is itself noisy
  mixed/<lang>+<other>/Alike         80 % <lang> with a 20 % block of
                                     <other> quoted in the middle (block
                                     boundaries in ``truth.sections``),
                                     one key from <lang>'s targets — POSITIVE
                                     with a foreign section

The B-like positive and the cross-language ("wrong hypothesis") cells reuse
``positive/<lang>/{Alike,Blike}`` from the Phase-6 wordhom controls.

Stages (artifacts under DATA_ROOT/analysis/wordhom/battery/):
  prepare   build the instances + manifest
  solve     CPU pool: inner n-gram search per (instance × hypothesis) →
            battery_solves.json (the loop's stuck starts)
  report    gather the loop's runs (analysis/altloop/runs<tag>.json) and the
            judge's verdicts (judge_at_ser<tag>.json) into one table

Usage:
  uv run python scripts/wordhom_battery.py --stage prepare
  uv run python scripts/wordhom_battery.py --stage solve --only /german --hyps german --workers 12
  uv run python scripts/altloop_pol.py --stage run --battery --cells shuffled/german/Alike:german ... --wild ...
  uv run python scripts/wordhom_battery.py --stage report --run-tags _bat_wild _bat_anneal --judge-tag _battery
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if "solve" in sys.argv or "prepare" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.ladder import write_json_atomic
from diff_voyn.vms.apply import (
    LANGS,
    WORDHOM,
    build_ngram_evaluator,
    instance_record,
    make_jobs,
    run_solves,
)

FAM = {"latin": "romance", "italian": "romance", "german": "germanic"}
SHAPES = {"Alike": (14000, 5200), "Blike": (30000, 7200)}
# the foreign block quoted inside each language's mixed instance
MIX_OTHER = {"german": "latin", "latin": "german", "italian": "latin"}
MIX_FRAC = 0.20
DIRTY_SEVERITIES = {"s05": 0.05, "s10": 0.10}


def battery_dir(root=None) -> Path:
    return (root or data_root()) / "analysis" / "wordhom" / "battery" / "wordtypesall"


def sample_long(sampler, length: int, rng) -> np.ndarray:
    """Held-out window of ``length`` letters from a doc long enough to hold it
    (``HeldoutSampler.sample`` assumes every doc is)."""
    ok = [i for i, d in enumerate(sampler.docs) if len(d) >= length]
    if not ok:
        raise ValueError(f"no held-out {sampler.language} doc of >= {length} letters")
    w = sampler.weights[ok] / sampler.weights[ok].sum()
    d = sampler.docs[ok[rng.choice(len(ok), p=w)]]
    start = rng.integers(0, len(d) - length + 1)
    return d[start : start + length].astype(np.int64)


def dirty_plain(plain: np.ndarray, severity: float, rng) -> tuple[np.ndarray, dict]:
    """Phase-2 transcription noise (sub 0.8 / del 0.1 / ins 0.1) on a letter
    stream in n-gram index space (0..24)."""
    from diff_voyn.data.noise import LETTER_BASE, N_LETTERS, TranscriptionNoise
    from diff_voyn.heads.ngram import A

    assert N_LETTERS == A
    ids = (plain + LETTER_BASE).astype(np.uint8)
    out, info = TranscriptionNoise(severity)(ids, rng)
    return out.astype(np.int64) - LETTER_BASE, info


# -- prepare -------------------------------------------------------------------


def stage_prepare(args):
    from diff_voyn.heads.synth import HeldoutSampler
    from diff_voyn.corpus.splits import load_splits
    from diff_voyn.heads.wordhom import language_targets
    from diff_voyn.vms.controls import (
        _letters_to_text,
        _rng,
        voynichesque_wordtypes_instance,
        wordhom_instance,
    )

    root = data_root()
    ev = build_ngram_evaluator()
    corpus_dir = root / "corpora" / "v1"
    splits = load_splits(corpus_dir)
    out_dir = battery_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    langs = args.langs or list(LANGS)

    def emit(control, inst):
        fname = re.sub(r"[^A-Za-z0-9_]+", "_", inst["name"]) + f"_{inst['kind']}.json"
        inst["control"] = control
        (out_dir / fname).write_text(json.dumps(inst))
        manifest.append(
            {
                "name": inst["name"],
                "kind": inst["kind"],
                "file": fname,
                "control": control,
                "truth": {
                    k: v
                    for k, v in inst["truth"].items()
                    if k not in ("plain_ids", "unit_ids", "sym_to_unit")
                },
                "coverage": inst["coverage"],
                "n_symbols": inst["n_symbols"],
                "n_stream": inst["n_stream"],
            }
        )
        print(
            f"{inst['name']:32s} {control:12s} types {inst['n_symbols']:5d} "
            f"tokens {inst['n_stream']:6d} ({inst['n_stream']/inst['n_symbols']:.1f}/type)",
            flush=True,
        )

    samplers = {l: HeldoutSampler(corpus_dir, splits, l) for l in LANGS}
    for lang in langs:
        smp = samplers[lang]
        targets = language_targets(ev, lang)
        base = {"language": lang, "family": FAM[lang], "in_inventory": True}
        for shape, (ln, nt) in SHAPES.items():
            # shuffled negative (same key generator as the positive)
            rng = _rng("battery-shuffled", args.seed, lang, shape)
            plain = rng.permutation(sample_long(smp, ln, rng))
            emit(
                "shuffled",
                wordhom_instance(
                    f"shuffled/{lang}/{shape}",
                    plain,
                    targets,
                    rng,
                    dict(base, shape=shape, source_language=lang),
                    n_types=nt,
                ),
            )
            # voynichesque negative at the shape's token count: the generator
            # yields ~0.5 tokens per source letter, so source ≈ 2 × tokens
            rng = _rng("battery-voynichesque", args.seed, lang, shape)
            n_tok_target = {"Alike": 13600, "Blike": 29200}[shape]
            tpt_target = {"Alike": 4.1, "Blike": 5.5}[shape]
            src_len = int(args.voyn_src_ratio * n_tok_target)
            src = sample_long(smp, src_len, rng)
            # the generator's parameter draw sets tokens/type (2.8–10 across
            # seeds): pick the draw closest to the shape's tokens/type, then
            # rescale the source to hit the token count
            seed0 = int(rng.integers(2**31))
            trials = []
            for k in range(args.voyn_trials):
                inst = voynichesque_wordtypes_instance(
                    f"voynichesque/{lang}/{shape}",
                    _letters_to_text(src),
                    seed0 + 1000 * k,
                    dict(base, shape=shape, source_language=lang),
                )
                trials.append((abs(inst["n_stream"] / inst["n_symbols"] - tpt_target), k, inst))
            _, k, inst = min(trials, key=lambda t: t[:2])
            scale = n_tok_target / inst["n_stream"]
            if abs(scale - 1) > 0.05:
                src2 = sample_long(smp, int(src_len * scale), _rng("battery-voynichesque", args.seed, lang, shape, "rescale"))
                inst = voynichesque_wordtypes_instance(
                    f"voynichesque/{lang}/{shape}",
                    _letters_to_text(src2),
                    seed0 + 1000 * k,
                    dict(base, shape=shape, source_language=lang),
                )
            inst["truth"]["voyn_trial"] = k
            emit("voynichesque", inst)
        # dirty positives (A-like)
        ln, nt = SHAPES["Alike"]
        for tag, sev in DIRTY_SEVERITIES.items():
            rng = _rng("battery-dirty", args.seed, lang, tag)
            clean = sample_long(smp, ln, rng)
            plain, info = dirty_plain(clean, sev, rng)
            emit(
                "dirty",
                wordhom_instance(
                    f"dirty/{lang}/Alike_{tag}",
                    plain,
                    targets,
                    rng,
                    dict(
                        base,
                        shape="Alike",
                        dirty=tag,
                        severity=sev,
                        noise_info={k: v for k, v in info.items()},
                        clean_ids=clean.tolist(),
                    ),
                    n_types=nt,
                ),
            )
        # mixed positive: lang | other-block | lang, one key from lang's targets
        other = MIX_OTHER[lang]
        rng = _rng("battery-mixed", args.seed, lang, other)
        n_other = int(round(MIX_FRAC * ln))
        n_a = (ln - n_other) // 2
        n_b = ln - n_other - n_a
        own = sample_long(smp, n_a + n_b, rng)
        blk = sample_long(samplers[other], n_other, rng)
        plain = np.concatenate([own[:n_a], blk, own[n_a:]])
        sections = [
            [0, n_a, lang],
            [n_a, n_a + n_other, other],
            [n_a + n_other, ln, lang],
        ]
        emit(
            "mixed",
            wordhom_instance(
                f"mixed/{lang}+{other}/Alike",
                plain,
                targets,
                rng,
                dict(
                    base,
                    shape="Alike",
                    other_language=other,
                    other_fraction=MIX_FRAC,
                    sections=sections,
                ),
                n_types=nt,
            ),
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "battery instances →", out_dir)


# -- solve ---------------------------------------------------------------------


def battery_paths(root=None):
    d = battery_dir(root)
    if not (d / "manifest.json").exists():
        return []
    return [d / m["file"] for m in json.loads((d / "manifest.json").read_text())]


def stage_solve(args):
    jobs = []
    for p in battery_paths():
        rec = instance_record(p)
        if args.only and not any(o in rec["name"] for o in args.only):
            continue
        js = make_jobs(
            rec,
            heads=(WORDHOM,),
            hypotheses=tuple(args.hyps),
            n_windows=1,
            w5=args.w5,
            restarts={WORDHOM: args.restarts},
        )
        for j in js:
            j["sa_steps"] = args.sa_steps
        jobs += js
    settings = {
        "w5": args.w5,
        "restarts": args.restarts,
        "sa_steps": args.sa_steps,
        "n_windows": 1,
    }
    run_solves(
        jobs,
        battery_dir().parent / "battery_solves.json",
        workers=args.workers,
        settings=settings,
        fresh=args.fresh,
    )


# -- report --------------------------------------------------------------------


def _section_ser(inst, final_map):
    """Letter SER inside each ``truth.sections`` block (mixed instances):
    the decode is aligned to the plaintext by the truth's unit boundaries, so
    a block's SER is measured on the tokens whose true units start in it."""
    from diff_voyn.heads.wordhom import UnitTargets, expand_units, unit_ser

    tr = inst["truth"]
    targets = UnitTargets.from_list(tr["bigrams"])
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    tm = np.asarray(tr["sym_to_unit"], dtype=np.int64)
    m = np.asarray(final_map, dtype=np.int64)
    true_units = tm[sym]
    ulen = 1 + (targets.second[true_units] >= 0)
    starts = np.concatenate([[0], np.cumsum(ulen)[:-1]])
    plain = np.asarray(tr["plain_ids"], dtype=np.int64)
    out = {}
    for s, e, lang in tr["sections"]:
        tok = np.flatnonzero((starts >= s) & (starts < e))
        if len(tok) == 0:
            continue
        dec = expand_units(m[sym[tok]], targets)
        ref = expand_units(true_units[tok], targets)
        out[f"{s}-{e}:{lang}"] = float(unit_ser(dec, ref))
    return out


def stage_report(args):
    root = data_root()
    bd = battery_dir(root)
    man = {m["name"]: m for m in json.loads((bd / "manifest.json").read_text())}
    runs = []
    for tag in args.run_tags:
        p = root / "analysis/altloop" / f"runs{tag}.json"
        if p.exists():
            for r in json.loads(p.read_text()):
                r["_tag"] = tag
                runs.append(r)
    judge = {}
    for tag in args.judge_tags:
        p = root / "analysis/altloop" / f"judge_at_ser{tag}.json"
        if p.exists():
            for r in json.loads(p.read_text()):
                judge[(r["cell"], r.get("hypothesis", r["truth_language"]), r["key"])] = r
    lines = [
        "# Wordhom battery — wildcard→anneal pipeline on manuscript-shaped controls",
        "",
        f"generated {datetime.now(UTC).isoformat()}; runs {args.run_tags}; judge {args.judge_tags}",
        "",
        "| cell (instance / hypothesis) | control | tok/type | stage | seed | SER start → final | obj Δ (nats) | rounds acc/n | judge: plain / margin / rank / called |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by = {}
    for r in runs:
        if not r["cell"].startswith("wh/"):
            continue
        _, *parts, hyp = r["cell"].split("/")
        name = "/".join(parts)
        by.setdefault((name, hyp), []).append(r)
    inst_cache = {}
    for (name, hyp), rs in sorted(by.items()):
        control = man.get(name, {}).get("control", "positive" if name.startswith("positive") else "?")
        mi = man.get(name)
        tpt = f"{mi['n_stream']/mi['n_symbols']:.1f}" if mi else ""
        for r in sorted(rs, key=lambda r: (r["_tag"], r["seed"])):
            sm, fm = r["start_metrics"], r["final_metrics"]
            f = lambda x: "–" if x is None else f"{x:.3f}"
            j = judge.get((name, hyp, f"final:{r['_tag']}/s{r['seed']}"))
            jtxt = ""
            if j:
                jtxt = (
                    f"{j['plain_bits']:.2f} / {j['structure_margin']:.2f} / "
                    f"{'>'.join(l[:2] for l in j['language_rank_of_decode'])} / "
                    f"{'YES' if j['called'] else 'no'}"
                )
            extra = ""
            if mi and "sections" in mi["truth"] and hyp == mi["truth"]["language"]:
                if name not in inst_cache:
                    inst_cache[name] = json.loads((bd / mi["file"]).read_text())
                ss = _section_ser(inst_cache[name], r["final_map"])
                extra = " sections: " + ", ".join(f"{k} {v:.3f}" for k, v in ss.items())
            lines.append(
                f"| {name} / {hyp} | {control} | {tpt} | {r['_tag']} | {r['seed']} | "
                f"{f(sm.get('ser'))} → {f(fm.get('ser'))}{extra} | {r['final_obj']-r['start_obj']:+.0f} | "
                f"{r['n_accepted']}/{r['n_rounds']} | {jtxt} |"
            )
    # judge rows for the reference keys (truth / stuck)
    ref = [(k, v) for k, v in judge.items() if k[2] in ("truth", "stuck")]
    if ref:
        lines += [
            "",
            "reference keys under the judge:",
            "",
            "| cell / hyp | key | SER | plain | margin | rank | top margin ± unc | like | called |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for (name, hyp, key), j in sorted(ref):
            lines.append(
                f"| {name} / {hyp} | {key} | {j['ser'] if j['ser'] is None else round(j['ser'],3)} | {j['plain_bits']:.2f} | "
                f"{j['structure_margin']:.2f} | {'>'.join(l[:2] for l in j['language_rank_of_decode'])} | "
                f"{j['top_margin_bits']:.3f} ± {j['top_margin_uncertainty_bits']:.3f} | "
                f"{'yes' if j['language_like'] else 'no'} | {'YES' if j['called'] else 'no'} |"
            )
    out = bd.parent / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", choices=["prepare", "solve", "report"], required=True)
    p.add_argument("--langs", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--voyn-src-ratio",
        type=float,
        default=1.85,
        help="source letters per target voynichesque token",
    )
    p.add_argument("--voyn-trials", type=int, default=12)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hyps", nargs="+", default=list(LANGS))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--w5", type=int, default=12000)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--sa-steps", type=int, default=2_000_000)
    p.add_argument("--run-tags", nargs="*", default=["_bat_wild", "_bat_anneal"])
    p.add_argument("--judge-tags", nargs="*", default=["_battery"])
    args = p.parse_args()
    {"prepare": stage_prepare, "solve": stage_solve, "report": stage_report}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
