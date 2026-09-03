"""Optimize the stacked sheet order for rare-type locality (2026-09-02).

Units: bifolia (sheets) as read stacked (leaf a r/v then leaf b r/v), with an
orientation flip allowed (b first).  Moves: swap two sheets that share a
constraint key, flip one sheet.  Objective: sum over consecutive occurrences
of types with 2..5 occurrences of exp(-gap/tau).  Reported: P(gap<=100) and
P(gap<=1000) over uniform, as in the corpus sweep.

Calibration: (a) the manuscript with pages shuffled within (quire, language)
before optimizing — the optimizer's gain on a text with no order information;
(b) known texts laid on the IT2a page slots, in a random sheet order within
the same constraint groups, optimized the same way; and their true-order values.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doubleton_gaps import DATA_ROOT, OUT, load_csv_text, load_raw_text, load_vms  # noqa: E402

TAU = 1000.0
KS = (2, 3, 4, 5)


def build_sheets(pages):
    sheets = defaultdict(list)
    for p in pages:
        sheets[(p["quire"], p["bifolio"])].append(p["page_idx"])
    qorder = list(dict.fromkeys(p["quire"] for p in pages))
    keys = sorted(sheets, key=lambda k: (qorder.index(k[0]), k[1]))
    units = []
    for k in keys:
        pidx = sheets[k]
        leaves = list(dict.fromkeys(pages[i]["page"].rstrip("0123456789").rstrip("rv") for i in pidx))
        # leaf a pages, leaf b pages (file order)
        a = [i for i in pidx if pages[i]["page"].rstrip("0123456789").rstrip("rv") == leaves[0]]
        b = [i for i in pidx if i not in a]
        maj = lambda f: Counter(pages[i][f] for i in pidx).most_common(1)[0][0]  # noqa: E731
        units.append({"key": k, "a": a, "b": b, "lang": maj("lang"), "section": maj("section"), "hand": maj("hand")})
    return units


class Objective:
    def __init__(self, words, page_of, n_pages):
        self.page_len = np.bincount(page_of, minlength=n_pages)
        start = np.concatenate([[0], np.cumsum(self.page_len)[:-1]])
        offset = np.arange(len(words)) - start[page_of]
        pos = defaultdict(list)
        for i, w in enumerate(words):
            pos[w].append(i)
        self.groups = {}
        for k in KS:
            idx = np.array([p for p in pos.values() if len(p) == k], dtype=int)
            if len(idx):
                self.groups[k] = (page_of[idx], offset[idx])
        self.n = len(words)
        rng = np.random.default_rng(0)
        self.e100 = self.e1000 = 0.0
        for k, (pg, _) in self.groups.items():
            U = np.sort(rng.choice(self.n, size=(len(pg) * 200, k)), axis=1)
            U = U[(np.diff(U, axis=1) > 0).all(axis=1)]
            ug = np.diff(U, axis=1).ravel()
            self.e100 += np.sum(ug <= 100) / 200
            self.e1000 += np.sum(ug <= 1000) / 200

    def gaps(self, page_order):
        new_start = np.zeros(len(self.page_len), dtype=np.int64)
        new_start[page_order] = np.concatenate([[0], np.cumsum(self.page_len[page_order])[:-1]])
        out = []
        for k, (pg, off) in self.groups.items():
            P = np.sort(new_start[pg] + off, axis=1)
            out.append(np.diff(P, axis=1).ravel())
        return np.concatenate(out)

    def value(self, page_order):
        return float(np.sum(np.exp(-self.gaps(page_order) / TAU)))

    def report(self, page_order):
        g = self.gaps(page_order)
        return {"J": float(np.sum(np.exp(-g / TAU))), "r100": float(np.sum(g <= 100) / self.e100), "r1000": float(np.sum(g <= 1000) / self.e1000)}


ORIENT = "none"  # "none": sheets always read a-r, a-v, b-r, b-v; "inverted": flip = b-v, b-r, a-r, a-v (sheet folded the other way)


def page_order_of(units, seq, flip):
    order = []
    for u_i in seq:
        u = units[u_i]
        if flip[u_i] and ORIENT == "inverted":
            order.extend(list(reversed(u["b"])) + u["a"])
        else:
            order.extend(u["a"] + u["b"])
    return np.array(order)


def anneal(obj, units, seq, flip, group_key, n_iter, rng, T0=2.0, T1=0.02):
    seq = list(seq); flip = list(flip)
    slots_by_group = defaultdict(list)
    for s_i, u_i in enumerate(seq):
        slots_by_group[group_key(units[u_i])].append(s_i)
    cur = obj.value(page_order_of(units, seq, flip)); best = cur; best_state = (list(seq), list(flip))
    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / max(1, n_iter - 1))
        if ORIENT != "none" and rng.random() < 0.3:
            u_i = rng.integers(len(units)); flip[u_i] ^= 1; undo = ("flip", u_i)
        else:
            gkeys = list(slots_by_group.keys())
            sl = slots_by_group[gkeys[rng.integers(len(gkeys))]]
            if len(sl) < 2:
                continue
            i, j = rng.choice(sl, 2, replace=False)
            seq[i], seq[j] = seq[j], seq[i]; undo = ("swap", i, j)
        new = obj.value(page_order_of(units, seq, flip))
        if new >= cur or rng.random() < np.exp((new - cur) / T):
            cur = new
            if cur > best:
                best = cur; best_state = (list(seq), list(flip))
        else:
            if undo[0] == "flip":
                flip[undo[1]] ^= 1
            else:
                seq[undo[1]], seq[undo[2]] = seq[undo[2]], seq[undo[1]]
    return best_state


LEVELS = {
    "strict (lang+section+hand)": lambda u: (u["lang"], u["section"], u["hand"]),
    "topic (lang+section)": lambda u: (u["lang"], u["section"]),
    "language only": lambda u: u["lang"],
    "free": lambda u: 0,
}


def run(words, page_of, units, n_pages, n_iter, rng, label, levels=LEVELS, init_shuffle=None):
    obj = Objective(words, page_of, n_pages)
    seq0 = list(range(len(units))); flip0 = [0] * len(units)
    base = obj.report(page_order_of(units, seq0, flip0))
    out = {"init": base}
    line = f"{label:34s} init r100 {base['r100']:.2f} r1000 {base['r1000']:.2f} |"
    for name, key in levels.items():
        seq, flip = seq0, flip0
        if init_shuffle:
            seq = list(seq0); flip = list(flip0)
            groups = defaultdict(list)
            for s_i, u_i in enumerate(seq):
                groups[key(units[u_i])].append(s_i)
            for sl in groups.values():
                vals = [seq[s] for s in sl]; rng.shuffle(vals)
                for s, v in zip(sl, vals):
                    seq[s] = v
            flip = [int(ORIENT != "none" and rng.random() < 0.5) for _ in units]
        bs, bf = anneal(obj, units, seq, flip, key, n_iter, rng)
        r = obj.report(page_order_of(units, bs, bf))
        out[name] = r
        line += f" {name.split(' ')[0]} r100 {r['r100']:.2f} r1000 {r['r1000']:.2f} |"
    print(line, flush=True)
    return out


def main():
    global ORIENT
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    ORIENT = sys.argv[2] if len(sys.argv) > 2 else "none"
    print(f"orientation moves: {ORIENT}")
    rng = np.random.default_rng(42)
    toks, pages = load_vms(DATA_ROOT / "raw" / "vms" / "IT2a-n.txt")
    words = [t["w"] for t in toks]; page_of = np.array([t["page_idx"] for t in toks])
    units = build_sheets(pages)
    print(f"{len(units)} sheets; constraint groups:", {n: len({k(u) for u in units}) for n, k in LEVELS.items()})
    res = {}
    res["vms_stacked"] = run(words, page_of, units, len(pages), n_iter, rng, "VMS from stacked order")
    res["vms_stacked_from_shuffled_sheets"] = [run(words, page_of, units, len(pages), n_iter, rng, f"VMS, sheets pre-shuffled #{i}", init_shuffle=True) for i in range(3)]
    # noise ceiling: page contents shuffled within (quire, lang) so no order information survives, then optimize
    groups = defaultdict(list)
    for p in pages:
        groups[(p["quire"], p["lang"])].append(p["page_idx"])
    res["vms_pages_shuffled"] = []
    for i in range(3):
        perm = np.arange(len(pages))
        for m in groups.values():
            m = np.array(m); perm[m] = rng.permutation(perm[m])
        res["vms_pages_shuffled"].append(run(words, perm[page_of], units, len(pages), n_iter, rng, f"VMS, page contents shuffled #{i}"))
    # known texts on the IT2a slots
    page_len = np.bincount(page_of, minlength=len(pages))
    cc = DATA_ROOT / "external/voynich-attack/corpora/latin/CorpusCorporum/auctores_scientiarum_varii"
    dta = DATA_ROOT / "external/voynich-attack/corpora/german/DTA"
    texts = {
        "la_isidorus_etym": load_csv_text(cc / "isidorus_hispalensis/etymologiae/etymologiae.csv"),
        "la_seneca_nq": load_csv_text(cc / "seneca/naturales_quaestiones/naturales_quaestiones.csv"),
        "de_bullinger": load_csv_text(dta / "1558_bullinger_haussbuoch/1558_bullinger_haussbuoch.csv"),
        "it_decameron": load_raw_text(DATA_ROOT / "raw/italian/boccaccio_decameron.txt"),
        "it_orlando_furioso": load_raw_text(DATA_ROOT / "raw/italian/ariosto_orlando_furioso.txt"),
    }
    res["known"] = {}
    stacked_order = page_order_of(units, list(range(len(units))), [0] * len(units))
    for name, src in texts.items():
        # write the text into the stacked order (so the stacked sequence is its true order)
        src = src[: int(page_len.sum())]
        kw, kp = [], []
        acc = 0
        slot_words = {}
        for slot in stacked_order:
            slot_words[int(slot)] = src[acc: acc + page_len[slot]]; acc += page_len[slot]
        for slot in range(len(pages)):
            kw.extend(slot_words[slot]); kp.extend([slot] * len(slot_words[slot]))
        kp = np.array(kp)
        r_true = run(kw, kp, units, len(pages), 0, rng, f"{name} true order (no opt)", levels={"free": lambda u: 0})
        r_shuf = [run(kw, kp, units, len(pages), n_iter, rng, f"{name} from shuffled sheets #{i}", init_shuffle=True) for i in range(2)]
        res["known"][name] = {"true": r_true, "from_shuffled": r_shuf}
    (OUT / f"order_optimize_{ORIENT}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
