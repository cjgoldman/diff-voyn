"""Doubleton gap study (side quest, 2026-09-02).

For every word type that occurs exactly twice, measure the distance in tokens
between its two occurrences, in the manuscript (folio order as bound) and in
known texts.  Compare against (a) the uniform-pair null (token shuffle) and
(b) page-order nulls (pages permuted globally / within Currier language /
within section / within quire; quires permuted as blocks), which is the
question raised by the "bound out of order" literature: does the current page
order carry information about where rare words recur?  The Davis "stacked
bifolia" reading order (each bifolium's four pages consecutive, outer to
inner) is evaluated as an explicit alternative order.

Outputs (DATA_ROOT/analysis/doubleton_gaps/):
  summary.json             — all statistics
  vms_<tr>_doubletons.csv  — every manuscript doubleton with both loci
  page_affinity_<tr>.csv   — shared-doubleton counts per page pair
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/workspace/data")
OUT = DATA_ROOT / "analysis" / "doubleton_gaps"
OUT.mkdir(parents=True, exist_ok=True)

_PAGE_RE = re.compile(r"^<(f[^.>]+)>\s*<!(.*)>\s*$")
_LOCUS_RE = re.compile(r"^<([^,>]+)\.([^,>]+),([^>]*)>\s*(.*)$")
_VAR_RE = re.compile(r"\$([A-Z])=(\S+)")
_ALT_RE = re.compile(r"\[([^:\]]*):[^\]]*\]")
_GAP_RE = re.compile(r"<->")
_MARKUP_RE = re.compile(r"<[^>]*>")
_EXT_RE = re.compile(r"@(\d+);")

STAT_KEYS = ("mean_log10_gap_cross", "frac_cross_le_1000", "frac_cross_le_3000", "frac_adjacent_page", "frac_within_3_pages", "frac_same_quire")


def load_vms(path: Path) -> tuple[list[dict], list[dict]]:
    """Tokens in folio order + page table.  Tokens with uncertain glyphs ('?')
    are dropped whole; extended-EVA codes and ligature braces are kept as part
    of the word type."""
    toks: list[dict] = []
    pages: list[dict] = []
    page = None
    dropped = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        m = _PAGE_RE.match(raw)
        if m and not _LOCUS_RE.match(raw):
            v = dict(_VAR_RE.findall(m.group(2)))
            page = {
                "page": m.group(1),
                "page_idx": len(pages),
                "quire": v.get("Q"),
                "bifolio": int(v.get("B", "0")),
                "lang": v.get("L", "?"),
                "section": v.get("I", "?"),
                "hand": v.get("H", "?"),
            }
            pages.append(page)
            continue
        lm = _LOCUS_RE.match(raw)
        if not (lm and page):
            continue
        text = lm.group(4)
        text = _ALT_RE.sub(r"\1", text)
        text = _GAP_RE.sub(".", text)
        text = _MARKUP_RE.sub("", text)
        text = _EXT_RE.sub(r"<\1>", text)
        text = text.replace("{", "").replace("}", "").replace("'", "")
        for w in re.split(r"[.,\s]+", text):
            if not w:
                continue
            if "?" in w or not re.fullmatch(r"[a-z<>0-9]+", w):
                dropped += 1
                continue
            toks.append({"w": w, "page_idx": page["page_idx"]})
    print(f"  {path.name}: {len(toks)} tokens on {len(pages)} pages, {dropped} dropped", file=sys.stderr)
    return toks, pages


def load_csv_text(path: Path) -> list[str]:
    words: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("block_type") not in ("body", "head"):
                continue
            words.extend(re.findall(r"[a-zäöüßàèéìòù]+", row["textstring_simple"].lower()))
    return words


def load_raw_text(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="replace").lower()
    return re.findall(r"[a-zàèéìíòóùú]+", txt)


# ----------------------------------------------------------------------------
def doubletons(words: list[str]) -> np.ndarray:
    pos: dict[str, list[int]] = defaultdict(list)
    for k, w in enumerate(words):
        pos[w].append(k)
    pairs = [(p[0], p[1]) for p in pos.values() if len(p) == 2]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def uniform_cdf(n: int, g: np.ndarray) -> np.ndarray:
    g = np.minimum(g, n - 1).astype(np.float64)
    return (n * g - g * (g + 1) / 2) / (n * (n - 1) / 2)


def gap_stats(n: int, gaps: np.ndarray) -> dict:
    gaps = np.asarray(gaps)
    out = {"n_tokens": n, "n_doubletons": int(len(gaps))}
    if len(gaps) == 0:
        return out
    out["median_gap"] = float(np.median(gaps))
    out["median_gap_over_n"] = float(np.median(gaps) / n)
    out["mean_log10_gap"] = float(np.mean(np.log10(gaps)))
    d = np.arange(1, n)
    pmf = (n - d) / (n * (n - 1) / 2)
    out["mean_log10_gap_uniform"] = float(np.sum(pmf * np.log10(d)))
    out["median_gap_uniform"] = float(d[np.searchsorted(np.cumsum(pmf), 0.5)])
    for g in (10, 30, 100, 300, 1000, 3000):
        obs = float(np.mean(gaps <= g))
        exp = float(uniform_cdf(n, np.array([g]))[0])
        out[f"frac_le_{g}"] = obs
        out[f"frac_le_{g}_uniform"] = exp
        out[f"ratio_le_{g}"] = obs / exp if exp > 0 else None
    return out


# ----------------------------------------------------------------------------
class OrderTest:
    """Doubleton statistics as a function of page order."""

    def __init__(self, words: list[str], page_of: np.ndarray, pages: list[dict]):
        self.n = len(words)
        self.pages = pages
        self.P = len(pages)
        self.page_of = page_of
        self.page_len = np.bincount(page_of, minlength=self.P)
        start = np.concatenate([[0], np.cumsum(self.page_len)[:-1]])
        self.offset = np.arange(self.n) - start[page_of]
        pairs = doubletons(words)
        self.pairs = pairs
        self.pi, self.pj = page_of[pairs[:, 0]], page_of[pairs[:, 1]]
        self.oi, self.oj = self.offset[pairs[:, 0]], self.offset[pairs[:, 1]]
        self.cross = self.pi != self.pj
        self.quire = np.array([p["quire"] for p in pages])
        self.n_cross = int(self.cross.sum())

    def stats(self, rank: np.ndarray) -> dict:
        """rank[p] = position of page p in the reading order (0..P-1)."""
        order = np.argsort(rank)
        new_start = np.zeros(self.P, dtype=np.int64)
        new_start[order] = np.concatenate([[0], np.cumsum(self.page_len[order])[:-1]])
        pi = new_start[self.pi] + self.oi
        pj = new_start[self.pj] + self.oj
        gaps = np.abs(pj - pi)[self.cross]
        dp = np.abs(rank[self.pi] - rank[self.pj])[self.cross]
        # quire of a page under a block permutation: quire label travels with the page
        sq = (self.quire[self.pi] == self.quire[self.pj])[self.cross]
        return {
            "mean_log10_gap_cross": float(np.mean(np.log10(gaps))),
            "frac_cross_le_1000": float(np.mean(gaps <= 1000)),
            "frac_cross_le_3000": float(np.mean(gaps <= 3000)),
            "frac_adjacent_page": float(np.mean(dp == 1)),
            "frac_within_3_pages": float(np.mean(dp <= 3)),
            "frac_same_quire": float(np.mean(sq)),
        }

    def null(self, kind: str, group_key, n_shuf: int, rng) -> dict:
        identity = np.arange(self.P)
        obs = self.stats(identity)
        groups = defaultdict(list)
        if kind == "page":
            for p in self.pages:
                groups[group_key(p)].append(p["page_idx"])
        null = defaultdict(list)
        # quire blocks in file order
        quire_order = list(dict.fromkeys(self.quire))
        quire_pages = {q: [p["page_idx"] for p in self.pages if p["quire"] == q] for q in quire_order}
        for _ in range(n_shuf):
            rank = identity.copy()
            if kind == "page":
                for members in groups.values():
                    members = np.array(members)
                    rank[members] = rng.permutation(rank[members])
            elif kind == "quire":
                acc = 0
                for q in rng.permutation(quire_order):
                    for p in quire_pages[q]:
                        rank[p] = acc
                        acc += 1
            for k, v in self.stats(rank).items():
                null[k].append(v)
        out = {"n_groups": len(groups) if kind == "page" else len(quire_order)}
        for k, v in obs.items():
            arr = np.array(null[k])
            sd = float(arr.std())
            hi_is_ordered = k != "mean_log10_gap_cross"
            out[k] = {
                "obs": v,
                "null_mean": float(arr.mean()),
                "null_sd": sd,
                "z": float((v - arr.mean()) / sd) if sd > 0 else None,
                "p_null_as_extreme": float(np.mean(arr >= v) if hi_is_ordered else np.mean(arr <= v)),
            }
        return out

    def alt_order(self, rank: np.ndarray, null_arrs: dict | None = None) -> dict:
        st = self.stats(rank)
        return st


def stacked_bifolia_rank(pages: list[dict]) -> np.ndarray:
    """Davis-style reading order: quires in current order; within a quire,
    bifolia from outermost ($B=1) inward, each bifolium's pages in file order
    (leaf a recto, verso, leaf b recto, verso)."""
    quire_order = list(dict.fromkeys(p["quire"] for p in pages))
    qrank = {q: k for k, q in enumerate(quire_order)}
    keyed = sorted(pages, key=lambda p: (qrank[p["quire"]], p["bifolio"], p["page_idx"]))
    rank = np.zeros(len(pages), dtype=np.int64)
    for r, p in enumerate(keyed):
        rank[p["page_idx"]] = r
    return rank


def run_order_tests(words, page_of, pages, n_shuf, seed=0, groupings=None) -> dict:
    rng = np.random.default_rng(seed)
    ot = OrderTest(words, page_of, pages)
    res = {"n_pages": ot.P, "n_doubletons": int(len(ot.pairs)), "n_cross_pairs": ot.n_cross,
           "frac_same_page": 1 - ot.n_cross / max(1, len(ot.pairs))}
    groupings = groupings or {"page_global": lambda p: "all"}
    for name, key in groupings.items():
        res[name] = ot.null("page", key, n_shuf, rng)
    res["quire_blocks"] = ot.null("quire", None, n_shuf, rng)
    if "bifolio" in pages[0]:
        res["stacked_bifolia_order"] = ot.stats(stacked_bifolia_rank(pages))
    return res


def pseudo_pages(n: int, page_len: int, quire_pages: int = 8) -> tuple[np.ndarray, list[dict]]:
    page_of = np.arange(n) // page_len
    P = int(page_of.max()) + 1
    pages = [{"page": f"p{p}", "page_idx": p, "quire": f"q{p // quire_pages}", "lang": "A", "section": "H"} for p in range(P)]
    return page_of, pages


def run_known(name: str, words: list[str], n_window: int, page_len: int, n_shuf: int) -> list[dict]:
    out = []
    n_windows = max(1, min(3, len(words) // n_window))
    for k in range(n_windows):
        w = words[k * n_window : (k + 1) * n_window]
        if len(w) < 8000:
            break
        pairs = doubletons(w)
        gaps = pairs[:, 1] - pairs[:, 0]
        st = gap_stats(len(w), gaps)
        st["types"] = len(set(w))
        st["hapax"] = sum(1 for c in Counter(w).values() if c == 1)
        page_of, pages = pseudo_pages(len(w), page_len)
        st["order"] = run_order_tests(w, page_of, pages, n_shuf, groupings={
            "page_global": lambda p: "all",
            "page_within_quire": lambda p: p["quire"],
        })
        st["name"] = f"{name}[{k}]"
        out.append(st)
        o = st["order"]
        print(f"  {st['name']}: n={len(w)} dbl={st['n_doubletons']} medgap {st['median_gap']:.0f}/{st['median_gap_uniform']:.0f} r100 {st['ratio_le_100']:.1f} samepage {o['frac_same_page']:.3f} adj z={o['page_global']['frac_adjacent_page']['z']:.1f} withinquire z={o['page_within_quire']['frac_adjacent_page']['z']:.1f}", file=sys.stderr)
    return out


def main() -> None:
    n_shuf = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    summary: dict = {"n_shuffles": n_shuf, "vms": {}, "known": {}}
    vms_groupings = {
        "page_global": lambda p: "all",
        "page_within_lang": lambda p: p["lang"],
        "page_within_section": lambda p: p["section"],
        "page_within_lang_section": lambda p: p["lang"] + p["section"],
        "page_within_quire": lambda p: p["quire"],
    }
    for tr, fname in (("IT2a", "IT2a-n.txt"), ("RF1b", "RF1b-e.txt")):
        toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / fname)
        words = [t["w"] for t in toks]
        page_of = np.array([t["page_idx"] for t in toks])
        n = len(words)
        pairs = doubletons(words)
        gaps = pairs[:, 1] - pairs[:, 0]
        st = gap_stats(n, gaps)
        st["types"] = len(set(words))
        st["hapax"] = sum(1 for c in Counter(words).values() if c == 1)
        st["mean_page_len"] = n / len(pages)
        pg = lambda k: pages[page_of[k]]  # noqa: E731
        st["frac_same_lang"] = float(np.mean([pg(i)["lang"] == pg(j)["lang"] for i, j in pairs]))
        st["frac_same_section"] = float(np.mean([pg(i)["section"] == pg(j)["section"] for i, j in pairs]))
        st["frac_same_bifolio"] = float(np.mean([(pg(i)["quire"], pg(i)["bifolio"]) == (pg(j)["quire"], pg(j)["bifolio"]) for i, j in pairs]))
        st["order"] = run_order_tests(words, page_of, pages, n_shuf, groupings=vms_groupings)
        for lang in ("A", "B"):
            keep = np.array([pages[p]["lang"] == lang for p in page_of])
            sw = [w for w, k in zip(words, keep) if k]
            sub_pages = [p for p in pages if p["lang"] == lang]
            remap = {p["page_idx"]: k for k, p in enumerate(sub_pages)}
            sub_pages = [{**p, "page_idx": remap[p["page_idx"]]} for p in sub_pages]
            sp_of = np.array([remap[p] for p, k in zip(page_of, keep) if k])
            sp = doubletons(sw)
            sst = gap_stats(len(sw), sp[:, 1] - sp[:, 0])
            sst["order"] = run_order_tests(sw, sp_of, sub_pages, n_shuf, groupings={
                "page_global": lambda p: "all",
                "page_within_section": lambda p: p["section"],
                "page_within_quire": lambda p: p["quire"],
            })
            st[f"currier_{lang}"] = sst
        summary["vms"][tr] = st
        o = st["order"]
        print(f"VMS {tr}: n={n} dbl={len(pairs)} medgap {st['median_gap']:.0f}/{st['median_gap_uniform']:.0f} r100 {st['ratio_le_100']:.2f} samepage {o['frac_same_page']:.3f} adj z={o['page_global']['frac_adjacent_page']['z']:.1f} withinquire z={o['page_within_quire']['frac_adjacent_page']['z']:.1f}", file=sys.stderr)

        with open(OUT / f"vms_{tr}_doubletons.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["word", "i", "j", "gap", "page_i", "page_j", "quire_i", "quire_j", "lang_i", "lang_j", "section_i", "section_j"])
            for (i, j), g in zip(pairs, gaps):
                a, b = pg(i), pg(j)
                wr.writerow([words[i], i, j, g, a["page"], b["page"], a["quire"], b["quire"], a["lang"], b["lang"], a["section"], b["section"]])
        aff: Counter = Counter()
        for i, j in pairs:
            if pg(i)["page"] != pg(j)["page"]:
                aff[(pg(i)["page"], pg(j)["page"])] += 1
        with open(OUT / f"page_affinity_{tr}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["page_a", "page_b", "shared_doubletons"])
            for (a, b), c in aff.most_common():
                wr.writerow([a, b, c])

    n_window = summary["vms"]["IT2a"]["n_tokens"]
    page_len = int(round(summary["vms"]["IT2a"]["mean_page_len"]))
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    known = {
        "la_macer_herbarum": load_csv_text(cc / "macer_floridus/de_viribus_herbarum/de_viribus_herbarum.csv"),
        "la_gordon_crisi": load_csv_text(cc / "bernardus_gordonensis/tractatus_de_crisi_et_de_diebus_creticis/tractatus_de_crisi_et_de_diebus_creticis.csv"),
        "la_hyginus_astron": load_csv_text(cc / "hyginus/de_astronomia/de_astronomia.csv"),
        "la_seneca_nq": load_csv_text(cc / "seneca/naturales_quaestiones/naturales_quaestiones.csv"),
        "la_isidorus_etym": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "la_plinius_nh": load_csv_text(cc / "plinius_maior/naturalis_historia/naturalis_historia.csv"),
        "la_baco_opus_majus": load_csv_text(cc / "rogerus_baco/opus_majus/opus_majus.csv"),
        "de_kuchemaistrey": load_csv_text(dta / "1490_nn_kuchemaistrey/1490_nn_kuchemaistrey.csv"),
        "de_promptuarium": load_csv_text(dta / "1483_nn_promptuarium/1483_nn_promptuarium.csv"),
        "de_bullinger_haussbuoch": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "de_staden_landschafft": load_csv_text(dta / "1557_staden_landschafft/1557_staden_landschafft.csv"),
        "it_principe": load_raw_text(DATA_ROOT / "raw/italian/machiavelli_il_principe.txt"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
        "it_commedia": load_raw_text(DATA_ROOT / "raw/italian/dante_divina_commedia.txt"),
        "it_orlando_furioso": load_raw_text(DATA_ROOT / "raw/italian/ariosto_orlando_furioso.txt"),
    }
    for name, words in known.items():
        print(f"{name}: {len(words)} words", file=sys.stderr)
        summary["known"][name] = run_known(name, words, n_window, page_len, n_shuf=max(200, n_shuf // 5))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT/'summary.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
