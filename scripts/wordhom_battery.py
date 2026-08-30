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
  revdouble/<lang>/Alike             cipher WITH the doubled-letter units but the
                                     hypothesis (truth.hyp_bigrams) letters-only —
                                     POSITIVE whose truth key is representable only
                                     as a projection (reverse of nodouble)
  bigram/<lang>/Alike                cipher over letters + top-5 doubled letters +
                                     top-20 non-doubled bigrams (unit spec d5b20,
                                     truth.cipher_units) — the matched POSITIVE of
                                     the doubles+bigrams decoder variant (run it
                                     under --units d5b20; under the default d5 it
                                     is the reverse mismatch, truth projected)
  nodouble/<lang>/Alike              plaintext enciphered WITHOUT doubled-letter
                                     units (every letter its own unit; a doubled
                                     letter is the same token twice under the
                                     repeat rule) while the HYPOTHESIS keeps the
                                     language's top-5 doubled units — POSITIVE
                                     whose decoder has 5 spare unit slots the
                                     cipher never uses (hypothesis mismatch)
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

The decoder's unit set is a run-time choice (``--units``, heads.wordhom
``parse_units``): ``d5`` (default) or ``d5b20`` (doubles + top-20 bigrams).
A non-default spec solves into ``battery_solves_<units>.json`` (positives from
the Phase-6 controls included when ``--only`` names them) and the loop/judge
read that file with the same ``--units``.

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
NODOUBLE_SHAPES = ("Alike",)
BIGRAM_UNITS = "d5b20"
CONTROLS = ("shuffled", "voynichesque", "dirty", "mixed", "nodouble", "revdouble", "bigram")


def battery_dir(root=None) -> Path:
    return (root or data_root()) / "analysis" / "wordhom" / "battery" / "wordtypesall"


# a held-out window whose own-language n-gram cross-entropy exceeds this is
# not usable plaintext: the Latin held-out set holds a pharmacopoeia (drug
# names, abbreviations, Roman-numeral doses) at 4.7 bits/char and 10 % of the
# sampling weight; ordinary docs score 2.2–3.2. Resample instead.
MAX_OWN_BPC = 3.6


def sample_long(sampler, length: int, rng, lm=None, max_bpc: float = MAX_OWN_BPC) -> np.ndarray:
    """Held-out window of ``length`` letters from a doc long enough to hold it
    (``HeldoutSampler.sample`` assumes every doc is); with ``lm`` (the
    language's n-gram LM) windows above ``max_bpc`` bits/char are redrawn."""
    ok = [i for i, d in enumerate(sampler.docs) if len(d) >= length]
    if not ok:
        raise ValueError(f"no held-out {sampler.language} doc of >= {length} letters")
    w = sampler.weights[ok] / sampler.weights[ok].sum()
    for _ in range(50):
        d = sampler.docs[ok[rng.choice(len(ok), p=w)]]
        start = rng.integers(0, len(d) - length + 1)
        win = d[start : start + length].astype(np.int64)
        if lm is None or (bpc := lm.bits_per_char(win)) <= max_bpc:
            return win
        print(f"  resample {sampler.language}: window at {bpc:.2f} bits/char > {max_bpc}", flush=True)
    raise ValueError(f"no {sampler.language} window under {max_bpc} bits/char in 50 draws")


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
    from diff_voyn.heads.ngram import A
    from diff_voyn.heads.wordhom import UnitTargets, language_targets, segment_units
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
    controls = set(args.controls or CONTROLS)

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
            plain = rng.permutation(sample_long(smp, ln, rng, ev.lms[lang]))
            if "shuffled" in controls:
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
            if "voynichesque" not in controls:
                continue
            # voynichesque negative at the shape's token count: the generator
            # yields ~0.5 tokens per source letter, so source ≈ 2 × tokens
            rng = _rng("battery-voynichesque", args.seed, lang, shape)
            n_tok_target = {"Alike": 13600, "Blike": 29200}[shape]
            tpt_target = {"Alike": 4.1, "Blike": 5.5}[shape]
            src_len = int(args.voyn_src_ratio * n_tok_target)
            src = sample_long(smp, src_len, rng, ev.lms[lang])
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
                src2 = sample_long(smp, int(src_len * scale), _rng("battery-voynichesque", args.seed, lang, shape, "rescale"), ev.lms[lang])
                inst = voynichesque_wordtypes_instance(
                    f"voynichesque/{lang}/{shape}",
                    _letters_to_text(src2),
                    seed0 + 1000 * k,
                    dict(base, shape=shape, source_language=lang),
                )
            inst["truth"]["voyn_trial"] = k
            emit("voynichesque", inst)
        # no-doubled-unit positives: the cipher has letter units only, the
        # hypothesis (and hence truth.bigrams, the decode's unit space) keeps
        # the language's doubled units; the letter-space truth key is valid
        # in that space, and doubled letters are repeat-rule token repeats
        for shape in NODOUBLE_SHAPES if "nodouble" in controls else ():
            ln, nt = SHAPES[shape]
            rng = _rng("battery-nodouble", args.seed, lang, shape)
            plain = sample_long(smp, ln, rng, ev.lms[lang])
            inst = wordhom_instance(
                f"nodouble/{lang}/{shape}",
                plain,
                UnitTargets(()),
                rng,
                dict(base, shape=shape, cipher_bigrams=[]),
                n_types=nt,
            )
            assert max(inst["truth"]["sym_to_unit"]) < A
            inst["truth"]["bigrams"] = targets.as_list()
            emit("nodouble", inst)
        # reverse mismatch: the cipher USES the doubled units, the hypothesis
        # (truth.hyp_bigrams) has letters only — the truth key is representable
        # only as its projection (doubled-unit types -> the base letter)
        for shape in NODOUBLE_SHAPES if "revdouble" in controls else ():
            ln, nt = SHAPES[shape]
            rng = _rng("battery-revdouble", args.seed, lang, shape)
            plain = sample_long(smp, ln, rng, ev.lms[lang])
            inst = wordhom_instance(
                f"revdouble/{lang}/{shape}",
                plain,
                targets,
                rng,
                dict(base, shape=shape),
                n_types=nt,
            )
            inst["truth"]["hyp_bigrams"] = []
            emit("revdouble", inst)
        # doubles + bigrams positive: the cipher's units are the d5b20 set
        # (letters, top-5 doubled letters, top-20 non-doubled bigrams); the
        # hypothesis is whatever --units the run uses (d5b20 = matched)
        for shape in NODOUBLE_SHAPES if "bigram" in controls else ():
            ln, nt = SHAPES[shape]
            rng = _rng("battery-bigram", args.seed, lang, shape)
            plain = sample_long(smp, ln, rng, ev.lms[lang])
            big = language_targets(ev, lang, units=BIGRAM_UNITS)
            assert big.bigrams[: len(targets.bigrams)] == targets.bigrams
            # bigram units absorb ~25 % of the letters into fewer tokens: scale
            # the key so tokens per type (the binding variable) matches the d5
            # cipher of the same plaintext rather than the letter count
            n_tok_big = len(segment_units(plain, big))
            n_tok_d5 = len(segment_units(plain, targets))
            nt_big = round(nt * n_tok_big / n_tok_d5)
            inst = wordhom_instance(
                f"bigram/{lang}/{shape}",
                plain,
                big,
                rng,
                dict(base, shape=shape, cipher_units=BIGRAM_UNITS, n_types_d5=nt),
                n_types=nt_big,
            )
            emit("bigram", inst)
        # dirty positives (A-like)
        ln, nt = SHAPES["Alike"]
        for tag, sev in DIRTY_SEVERITIES.items() if "dirty" in controls else ():
            rng = _rng("battery-dirty", args.seed, lang, tag)
            clean = sample_long(smp, ln, rng, ev.lms[lang])
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
        if "mixed" not in controls:
            continue
        other = MIX_OTHER[lang]
        rng = _rng("battery-mixed", args.seed, lang, other)
        n_other = int(round(MIX_FRAC * ln))
        n_a = (ln - n_other) // 2
        n_b = ln - n_other - n_a
        own = sample_long(smp, n_a + n_b, rng, ev.lms[lang])
        blk = sample_long(samplers[other], n_other, rng, ev.lms[other])
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
    mp = out_dir / "manifest.json"
    if mp.exists():  # merge: rebuilt instances replace their old entries
        new = {m["name"] for m in manifest}
        manifest = [m for m in json.loads(mp.read_text()) if m["name"] not in new] + manifest
    mp.write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "battery instances →", out_dir)


# -- solve ---------------------------------------------------------------------


def battery_paths(root=None):
    d = battery_dir(root)
    if not (d / "manifest.json").exists():
        return []
    return [d / m["file"] for m in json.loads((d / "manifest.json").read_text())]


def control_positive_paths(root=None):
    """The Phase-6 wordhom controls' positive/* instances (reused by the
    battery as B-like positives and cross-language cells)."""
    d = (root or data_root()) / "analysis/wordhom/controls/wordtypesall"
    if not (d / "manifest.json").exists():
        return []
    return [
        d / m["file"]
        for m in json.loads((d / "manifest.json").read_text())
        if m["name"].startswith("positive/")
    ]


def stage_solve(args):
    from diff_voyn.heads.wordhom import units_suffix

    jobs = []
    # positives are solved here only when --only names them explicitly (the
    # default unit set already has them in controls_solves.json)
    paths = battery_paths() + (control_positive_paths() if args.only else [])
    for p in paths:
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
            units=args.units,
        )
        for j in js:
            j["sa_steps"] = args.sa_steps
        jobs += js
    settings = {
        "w5": args.w5,
        "restarts": args.restarts,
        "sa_steps": args.sa_steps,
        "n_windows": 1,
        "units": args.units,
    }
    run_solves(
        jobs,
        battery_dir().parent / f"battery_solves{units_suffix(args.units)}.json",
        workers=args.workers,
        settings=settings,
        fresh=args.fresh,
    )


# -- report --------------------------------------------------------------------


def _section_ser(inst, final_map, hyp_targets=None):
    """Letter SER inside each ``truth.sections`` block (mixed instances):
    the decode is aligned to the plaintext by the truth's unit boundaries, so
    a block's SER is measured on the tokens whose true units start in it.
    ``hyp_targets`` is the decoder's unit space when it differs from the
    cipher's (``final_map`` lives there)."""
    from diff_voyn.heads.wordhom import UnitTargets, expand_units, unit_ser

    tr = inst["truth"]
    targets = UnitTargets.from_list(tr["bigrams"])
    hyp_targets = hyp_targets or targets
    sym = np.asarray(inst["symbols"], dtype=np.int64)
    tm = np.asarray(tr["sym_to_unit"], dtype=np.int64)
    m = np.asarray(final_map, dtype=np.int64)
    true_units = tm[sym]
    ulen = 1 + (targets.second[true_units] >= 0)
    starts = np.concatenate([[0], np.cumsum(ulen)[:-1]])
    out = {}
    for s, e, lang in tr["sections"]:
        tok = np.flatnonzero((starts >= s) & (starts < e))
        if len(tok) == 0:
            continue
        dec = expand_units(m[sym[tok]], hyp_targets)
        ref = expand_units(true_units[tok], targets)
        out[f"{s}-{e}:{lang}"] = float(unit_ser(dec, ref))
    return out


def stage_report(args):
    from diff_voyn.heads.wordhom import hypothesis_targets

    root = data_root()
    bd = battery_dir(root)
    ng = build_ngram_evaluator() if args.units else None
    man = {m["name"]: m for m in json.loads((bd / "manifest.json").read_text())}
    cm = root / "analysis/wordhom/controls/wordtypesall/manifest.json"
    if cm.exists():  # positive/* cells are reused from the controls set
        for m in json.loads(cm.read_text()):
            man.setdefault(m["name"], m)
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
        "# Wordhom battery — wildcard→anneal pipeline on manuscript-shaped controls"
        + (f" (decoder unit set `{args.units}`)" if args.units else ""),
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
            if mi and len(r["final_map"]) != mi["n_symbols"]:
                # run on a since-rebuilt instance (different key size): stale
                lines.append(f"| {name} / {hyp} | {control} | {tpt} | {r['_tag']} | {r['seed']} | (stale: instance rebuilt) | | | |")
                continue
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
                ht = (
                    hypothesis_targets(ng, hyp, units=args.units, inst=inst_cache[name])
                    if ng is not None
                    else None
                )
                ss = _section_ser(inst_cache[name], r["final_map"], ht)
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
    out = bd.parent / (args.report_name or "report.md")
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
    p.add_argument("--controls", nargs="*", default=None, choices=CONTROLS,
                   help="prepare only these controls (merged into the manifest)")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--hyps", nargs="+", default=list(LANGS))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--w5", type=int, default=12000)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--sa-steps", type=int, default=2_000_000)
    p.add_argument("--run-tags", nargs="*", default=["_bat_wild", "_bat_anneal"])
    p.add_argument("--judge-tags", nargs="*", default=["_battery"])
    p.add_argument(
        "--units",
        default=None,
        help="decoder unit-set spec (d5 default, d5b20 = doubles + top-20 bigrams): "
        "solve writes battery_solves_<units>.json; report labels the table",
    )
    p.add_argument("--report-name", default=None, help="report file name (default report.md)")
    args = p.parse_args()
    {"prepare": stage_prepare, "solve": stage_solve, "report": stage_report}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
