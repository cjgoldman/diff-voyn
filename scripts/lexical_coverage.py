"""Human-shaped call criterion: in-lexicon character coverage of a decode.

For a decoded letter stream (no spaces) and a language's word list built
from the TRAINING split only, a DP segments the stream to maximise the
number of characters covered by lexicon words of length >= L_min (default
4; shorter words cover random text too easily). Reported per hypothesis
lexicon; the "lexical margin" is coverage(decode) - coverage(letter-shuffled
decode), the analogue of the structure margin.

Evaluated on: the A-like wordhom cells (truth / stuck / anneal finals /
uniform + rare-first corruptions of the truth), every wordhom control
(n-gram MDL pick under each hypothesis), and the Phase-6 control solves.
Artifacts: DATA_ROOT/analysis/altloop/lexical_coverage.{json,md},
lexicons cached under DATA_ROOT/analysis/lexicon/.
"""
from __future__ import annotations
import json, re, sys, argparse
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from diff_voyn.ciphers.external import data_root
from diff_voyn.corpus.assemble import _read_va_csv
from diff_voyn.normalize import normalize
from diff_voyn.vocab import LETTERS
from diff_voyn.heads.wordhom import UnitTargets, expand_units, unit_ser
from judge_at_ser import corrupt

ROOT = data_root()
LANGS = ["latin", "italian", "german"]
LMAX = 20


def build_counts(lang: str) -> Counter:
    cache = ROOT / "analysis/lexicon" / f"{lang}_train_counts.json"
    if cache.exists():
        return Counter(json.loads(cache.read_text()))
    man = json.loads((ROOT / "corpora/v1/manifest.json").read_text())["documents"][lang]
    splits = json.loads((ROOT / "corpora/v1/splits_v1.json").read_text())["languages"][lang]
    train_ids = {d["doc_id"] for d in splits["train"]}
    cnt = Counter()
    for d in man:
        if d["doc_id"] not in train_ids:
            continue
        p = Path(d["source_path"])
        raw = _read_va_csv(p) if p.suffix == ".csv" else p.read_text(encoding="utf-8", errors="ignore")
        for w in re.findall(r"[^\W\d_]+", raw):
            nw = normalize(w)
            if nw:
                cnt[nw] += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dict(cnt.most_common(400000))))
    return cnt


def build_lexicon(lang: str, topk: int, lmin: int) -> set[str]:
    """The ``topk`` most frequent training-split words of length >= lmin —
    equal-size lexicons across languages (fairness) so coverage is comparable."""
    cnt = build_counts(lang)
    return {w for w, _ in [(w, c) for w, c in cnt.most_common() if len(w) >= lmin][:topk]}


def coverage(s: str, lex: set[str], lmin: int) -> float:
    n = len(s)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        best = dp[i - 1]
        for L in range(lmin, min(LMAX, i) + 1):
            if s[i - L:i] in lex:
                v = dp[i - L] + L
                if v > best:
                    best = v
        dp[i] = best
    return dp[n] / max(n, 1)


def ids_to_str(ids) -> str:
    return "".join(LETTERS[i] for i in ids)


def evaluate(ids, lexs, lmin, rng) -> dict:
    s = ids_to_str(ids)
    sh = ids_to_str(rng.permutation(np.asarray(ids)))
    out = {}
    for l in LANGS:
        c = coverage(s, lexs[l], lmin); cs = coverage(sh, lexs[l], lmin)
        out[l] = {"cov": round(c, 4), "cov_shuf": round(cs, 4), "lex_margin": round(c - cs, 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmin", type=int, default=5)
    ap.add_argument("--topk", type=int, default=10000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-chars", type=int, default=6000, help="truncate decodes for speed")
    args = ap.parse_args()
    lexs = {l: build_lexicon(l, args.topk, args.lmin) for l in LANGS}
    print({l: len(v) for l, v in lexs.items()})
    rng = np.random.default_rng(0)
    res = []

    def add(group, cell, key, hyp, truth_lang, ids, ser=None):
        ids = np.asarray(ids)[: args.max_chars]
        r = {"group": group, "cell": cell, "key": key, "hyp": hyp, "truth_language": truth_lang,
             "ser": ser, "n": int(len(ids)), "by_lang": evaluate(ids, lexs, args.lmin, rng)}
        bl = r["by_lang"]
        r["cov_hyp"] = bl[hyp]["cov"] if hyp in bl else None
        r["lex_margin_hyp"] = bl[hyp]["lex_margin"] if hyp in bl else None
        r["top_lang"] = max(LANGS, key=lambda l: bl[l]["cov"])
        res.append(r)
        print(f"{group:10s} {cell:28s} {key:16s} hyp={hyp:8s} ser={'-' if ser is None else f'{ser:.3f}'} "
              f"cov={ {l: bl[l]['cov'] for l in LANGS} } top={r['top_lang']}", flush=True)

    # --- A-like wordhom cells ---
    wd = ROOT / "analysis/wordhom"
    solves = json.loads((wd / "controls_solves.json").read_text())["instances"]
    anneal = {}
    for fn in ["runs_anneal.json", "runs_anneal_de.json"]:
        for r in json.loads((ROOT / "analysis/altloop" / fn).read_text()):
            anneal.setdefault(r["cell"], []).append((r["seed"], r["final_map"]))
    for lang in LANGS:
        name = f"positive/{lang}/Alike"
        inst = json.loads((wd / "controls/wordtypesall" / (name.replace("/", "_") + "_wordtypesall.json")).read_text())
        tr = inst["truth"]; true_map = np.asarray(tr["sym_to_unit"]); plain = np.asarray(tr["plain_ids"])
        targets = UnitTargets.from_list(tr["bigrams"]); sym = np.asarray(inst["symbols"])
        occ = np.bincount(sym, minlength=len(true_map))
        def dec_of(m): return expand_units(np.asarray(m)[sym], targets)
        keys = {"truth": true_map}
        rec = next(s for s in solves if s["instance"] == name and s["hypothesis"] == lang)
        keys["stuck"] = np.asarray(max(rec["candidates"], key=lambda c: c["inner"])["map"])
        for sd, m in anneal.get(f"wh/{name}/{lang}", []):
            keys[f"anneal/s{sd}"] = np.asarray(m)
        crng = np.random.default_rng(1000)
        for f in [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]:
            keys[f"uni@{f:.2f}"] = corrupt(true_map, occ, f, crng, targets.n, False)
        for f in [0.30, 0.55, 0.75]:
            keys[f"rare@{f:.2f}"] = corrupt(true_map, occ, f, crng, targets.n, True)
        for k, m in keys.items():
            d = dec_of(m)
            add("Alike", name, k, lang, lang, d, ser=float(unit_ser(d[: 4000], plain[: 4000])))

    # --- all wordhom controls: n-gram MDL pick per hypothesis ---
    for s in solves:
        if "Alike" in s["instance"]:
            continue
        best = max(s["candidates"], key=lambda c: c["inner"])
        grp = s["instance"].split("/")[0]
        tl = s["instance"].split("/")[1]
        add("wh-" + grp, s["instance"], "mdlpick", s["hypothesis"], tl, best["decode"])

    # --- Phase-6 control solves (naibbe: stored decode; sub1to1/homophonic: map[symbols]) ---
    def p6_decode(s, inst_dir):
        best = max(s["candidates"], key=lambda c: c["inner"])
        if best.get("decode"):
            return np.asarray(best["decode"]), best
        fn = inst_dir / (s["instance"].replace("/", "_") + f"_{s['presentation']}.json")
        if not fn.exists():
            return None, best
        inst = json.loads(fn.read_text())
        sym = np.asarray(inst["symbols"])
        m = np.asarray(best["map"])
        a, b = s["window_span"]
        return m[sym[a:b]], best
    p6 = json.loads((ROOT / "analysis/phase6/controls/solves.json").read_text())["instances"]
    for s in p6:
        d, best = p6_decode(s, ROOT / "analysis/phase6/controls")
        if d is None:
            continue
        grp = s["instance"].split("/")[0]; tl = s["instance"].split("/")[1]
        ser = None
        if grp == "positive" and s["head"] == "naibbe" and s["hypothesis"] == tl:
            inst = json.loads((ROOT / "analysis/phase6/controls" / (s["instance"].replace("/", "_") + "_words.json")).read_text())
            pl = np.asarray(inst["truth"]["plain_ids"]); n = min(len(pl), len(d), 2000)
            ser = float(unit_ser(d[:n], pl[:n]))
        add("p6-" + grp, f"{s['instance']}/{s['head']}", "pick", s["hypothesis"], tl, d, ser=ser)

    # --- the manuscript: wordhom cells and Phase-6 character/naibbe heads ---
    for s in json.loads((ROOT / "analysis/wordhom/vms_solves.json").read_text())["instances"]:
        best = max(s["candidates"], key=lambda c: c["inner"])
        add("VMS-wordhom", s["instance"], "mdlpick", s["hypothesis"], "vms", best["decode"])
    for s in json.loads((ROOT / "analysis/phase6/vms_solves.json").read_text())["instances"]:
        d, best = p6_decode(s, ROOT / "analysis/phase6/presentations")
        if d is None:
            continue
        add("VMS-" + s["head"], f"{s['instance']}/{s['presentation']}", "pick", s["hypothesis"], "vms", d)

    out = ROOT / "analysis/altloop" / f"lexical_coverage{args.tag}.json"
    out.write_text(json.dumps({"lmin": args.lmin, "topk": args.topk, "lexicon_sizes": {l: len(v) for l, v in lexs.items()}, "results": res}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
