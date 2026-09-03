"""Leaf-pair affinity of manuscript doubletons (side quest, 2026-09-02).

Question (Davis): were the bifolia meant to be stacked (each sheet read
a-recto, a-verso, b-recto, b-verso) rather than nested into quires?  If text
ran continuously across a sheet, conjugate leaves (same bifolium) should share
more rare words than leaves that are neighbours only in the nested order.

For every pair of leaves (folios) in the same quire we count doubletons with
one occurrence on each leaf, in categories:
  conjugate        same bifolium, different leaf            (adjacent only if stacked)
  nested_adjacent  consecutive folios, different bifolium   (adjacent only if nested)
  both             innermost bifolium (conjugate AND consecutive)
  other            neither
Null: permute page contents among the physical page slots of the same quire
(and same Currier language), which keeps the quire membership and the
language mix fixed and makes all leaf pairs exchangeable.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, doubletons, load_vms  # noqa: E402

_FOLIO_RE = re.compile(r"^(f\d+)([rv])(\d*)$")


def leaf_of(page: str) -> str:
    m = _FOLIO_RE.match(page)
    return m.group(1) if m else page


def analyse(tr: str, fname: str, n_shuf: int, rng, lang_filter: str | None = None) -> dict:
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
    for p in pages:
        p["leaf"] = leaf_of(p["page"])
    words = [t["w"] for t in toks]
    page_of = np.array([t["page_idx"] for t in toks])
    pairs = doubletons(words)
    pi, pj = page_of[pairs[:, 0]], page_of[pairs[:, 1]]
    # leaf categories per (page slot a, page slot b)
    quire_pages = defaultdict(list)
    for p in pages:
        quire_pages[p["quire"]].append(p)
    cat_of_slots: dict[tuple[int, int], str] = {}
    for q, plist in quire_pages.items():
        leaves = list(dict.fromkeys(p["leaf"] for p in plist))
        leaf_rank = {lf: k for k, lf in enumerate(leaves)}
        leaf_bif = {p["leaf"]: p["bifolio"] for p in plist}
        for a in plist:
            for b in plist:
                if a["page_idx"] >= b["page_idx"] or a["leaf"] == b["leaf"]:
                    continue
                conj = leaf_bif[a["leaf"]] == leaf_bif[b["leaf"]]
                adj = abs(leaf_rank[a["leaf"]] - leaf_rank[b["leaf"]]) == 1
                cat = "both" if (conj and adj) else "conjugate" if conj else "nested_adjacent" if adj else "other"
                cat_of_slots[(a["page_idx"], b["page_idx"])] = cat
                cat_of_slots[(b["page_idx"], a["page_idx"])] = cat
    cats = ("conjugate", "nested_adjacent", "both", "other")
    n_slot_pairs = Counter(cat_of_slots[k] for k in cat_of_slots if k[0] < k[1])

    def count(slot_of_page: np.ndarray) -> dict:
        c = Counter()
        for a, b in zip(slot_of_page[pi], slot_of_page[pj]):
            cat = cat_of_slots.get((int(a), int(b)))
            if cat:
                c[cat] += 1
        return {k: c.get(k, 0) for k in cats}

    identity = np.arange(len(pages))
    obs = count(identity)
    # permutation null: page contents shuffled among slots of the same quire+language
    groups = defaultdict(list)
    for p in pages:
        if lang_filter is None or p["lang"] == lang_filter:
            groups[(p["quire"], p["lang"])].append(p["page_idx"])
    null = defaultdict(list)
    for _ in range(n_shuf):
        perm = identity.copy()
        for members in groups.values():
            m = np.array(members)
            perm[m] = rng.permutation(perm[m])
        for k, v in count(perm).items():
            null[k].append(v)
        null["conj_minus_nested"].append(null["conjugate"][-1] - null["nested_adjacent"][-1])
    obs["conj_minus_nested"] = obs["conjugate"] - obs["nested_adjacent"]
    out = {"n_doubletons": int(len(pairs)), "n_slot_pairs": dict(n_slot_pairs)}
    for k in list(cats) + ["conj_minus_nested"]:
        arr = np.array(null[k], dtype=float)
        sd = arr.std()
        out[k] = {
            "obs": int(obs[k]),
            "null_mean": float(arr.mean()),
            "null_sd": float(sd),
            "z": float((obs[k] - arr.mean()) / sd) if sd > 0 else None,
            "p_ge": float(np.mean(arr >= obs[k])),
        }
    return out


def main() -> None:
    n_shuf = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(1)
    res = {}
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        res[tr] = analyse(tr, fname, n_shuf, rng)
        r = res[tr]
        print(f"== {tr}: doubletons {r['n_doubletons']}, leaf-pair slots {r['n_slot_pairs']}")
        for k in ("conjugate", "nested_adjacent", "both", "other", "conj_minus_nested"):
            x = r[k]
            print(f"   {k:18s} obs {x['obs']:4d}  null {x['null_mean']:6.1f} ± {x['null_sd']:4.1f}  z={x['z']:+.2f}  p(null>=obs)={x['p_ge']:.3f}")
    (OUT / "leaf_affinity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__" and "--control" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# Power control: a known text laid onto the manuscript's own page slots
# (same token count per page, same quire / bifolium structure), written either
# in the nested order (as bound) or in the stacked order (Davis), then analysed
# exactly like the manuscript.
def control(words_src: list[str], n_shuf: int, rng, mode: str, tr_file: str = "IT2a-n.txt") -> dict:
    from doubleton_gaps import stacked_bifolia_rank

    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / tr_file)
    for p in pages:
        p["leaf"] = leaf_of(p["page"])
    page_of = np.array([t["page_idx"] for t in toks])
    page_len = np.bincount(page_of, minlength=len(pages))
    if mode == "nested":
        rank = np.arange(len(pages))
    else:
        rank = stacked_bifolia_rank(pages)
    order = np.argsort(rank)  # page slots in writing order
    src = list(words_src[: int(page_len.sum())])
    if len(src) < page_len.sum():
        raise ValueError("control text too short")
    slot_words = {}
    acc = 0
    for slot in order:
        slot_words[int(slot)] = src[acc : acc + page_len[slot]]
        acc += page_len[slot]
    words = []
    new_page_of = []
    for slot in range(len(pages)):
        words.extend(slot_words[slot])
        new_page_of.extend([slot] * len(slot_words[slot]))
    return _analyse_words(words, np.array(new_page_of), pages, n_shuf, rng)


def _analyse_words(words, page_of, pages, n_shuf, rng) -> dict:
    """Same as analyse() but on prepared tokens (shared core)."""
    pairs = doubletons(words)
    pi, pj = page_of[pairs[:, 0]], page_of[pairs[:, 1]]
    quire_pages = defaultdict(list)
    for p in pages:
        quire_pages[p["quire"]].append(p)
    cat_of_slots = {}
    for q, plist in quire_pages.items():
        leaves = list(dict.fromkeys(p["leaf"] for p in plist))
        leaf_rank = {lf: k for k, lf in enumerate(leaves)}
        leaf_bif = {p["leaf"]: p["bifolio"] for p in plist}
        for a in plist:
            for b in plist:
                if a["page_idx"] >= b["page_idx"] or a["leaf"] == b["leaf"]:
                    continue
                conj = leaf_bif[a["leaf"]] == leaf_bif[b["leaf"]]
                adj = abs(leaf_rank[a["leaf"]] - leaf_rank[b["leaf"]]) == 1
                cat = "both" if (conj and adj) else "conjugate" if conj else "nested_adjacent" if adj else "other"
                cat_of_slots[(a["page_idx"], b["page_idx"])] = cat
                cat_of_slots[(b["page_idx"], a["page_idx"])] = cat
    cats = ("conjugate", "nested_adjacent", "both", "other")

    def count(slot_of_page):
        c = Counter()
        for a, b in zip(slot_of_page[pi], slot_of_page[pj]):
            cat = cat_of_slots.get((int(a), int(b)))
            if cat:
                c[cat] += 1
        return {k: c.get(k, 0) for k in cats}

    identity = np.arange(len(pages))
    obs = count(identity)
    groups = defaultdict(list)
    for p in pages:
        groups[(p["quire"], p["lang"])].append(p["page_idx"])
    null = defaultdict(list)
    for _ in range(n_shuf):
        perm = identity.copy()
        for members in groups.values():
            m = np.array(members)
            perm[m] = rng.permutation(perm[m])
        c = count(perm)
        for k, v in c.items():
            null[k].append(v)
        null["conj_minus_nested"].append(c["conjugate"] - c["nested_adjacent"])
    obs["conj_minus_nested"] = obs["conjugate"] - obs["nested_adjacent"]
    out = {"n_doubletons": int(len(pairs))}
    for k in list(cats) + ["conj_minus_nested"]:
        arr = np.array(null[k], dtype=float)
        sd = arr.std()
        out[k] = {"obs": int(obs[k]), "null_mean": float(arr.mean()), "null_sd": float(sd),
                  "z": float((obs[k] - arr.mean()) / sd) if sd > 0 else None, "p_ge": float(np.mean(arr >= obs[k]))}
    return out


def run_controls(n_shuf: int) -> None:
    from doubleton_gaps import load_csv_text, load_raw_text

    rng = np.random.default_rng(2)
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus_etym": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "la_seneca_nq": load_csv_text(cc / "seneca/naturales_quaestiones/naturales_quaestiones.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
        "it_orlando_furioso": load_raw_text(DATA_ROOT / "raw/italian/ariosto_orlando_furioso.txt"),
    }
    res = {}
    for name, words in texts.items():
        for mode in ("nested", "stacked"):
            r = control(words, n_shuf, rng, mode)
            res[f"{name}/{mode}"] = r
            print(f"== control {name} written {mode}: doubletons {r['n_doubletons']}")
            for k in ("conjugate", "nested_adjacent", "both", "other", "conj_minus_nested"):
                x = r[k]
                print(f"   {k:18s} obs {x['obs']:4d}  null {x['null_mean']:6.1f} ± {x['null_sd']:4.1f}  z={x['z']:+.2f}")
    (OUT / "leaf_affinity_controls.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--control":
    run_controls(int(sys.argv[1]))
