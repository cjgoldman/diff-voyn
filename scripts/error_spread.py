"""Spatial distribution of letter errors in the latest A-like wordhom finals.

Per covered token position, wrong iff its type is mis-mapped (map[sym] !=
truth[sym]); errors are expanded to letters. Compares window error rates,
correct-run lengths and the judge's 1024-char scoring windows against an iid
Bernoulli shuffle at the same overall rate.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.wordhom import UnitTargets, expand_units, unit_ser

root = data_root()
W = 1024
CELLS = [("positive/german/Alike", "german", "runs_anneal_de.json"),
         ("positive/italian/Alike", "italian", "runs_anneal.json"),
         ("positive/latin/Alike", "latin", "runs_anneal.json")]

def runs_of(x):
    # lengths of maximal runs of True
    x = np.concatenate([[0], x.astype(int), [0]])
    d = np.diff(x); s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    return e - s

out = []
for name, lang, fn in CELLS:
    inst = json.loads((root / "analysis/wordhom/controls/wordtypesall" / (name.replace("/", "_") + "_wordtypesall.json")).read_text())
    tr = inst["truth"]; true_map = np.asarray(tr["sym_to_unit"]); plain = np.asarray(tr["plain_ids"])
    targets = UnitTargets.from_list(tr["bigrams"]); sym = np.asarray(inst["symbols"])
    occ = np.bincount(sym, minlength=len(true_map))
    for r in json.loads((root / "analysis/altloop" / fn).read_text()):
        if r["cell"] != f"wh/{name}/{lang}": continue
        m = np.asarray(r["final_map"]); dec = expand_units(m[sym], targets)
        ser = unit_ser(dec, plain)
        wrong_tok = (m != true_map)[sym]
        # expand token-level flag to letters emitted by the *decode* (bigram units emit 2)
        nlet = 1 + (targets.second[m[sym]] >= 0)
        wrong_let = np.repeat(wrong_tok, nlet)
        n = len(wrong_let); p = wrong_let.mean()
        rng = np.random.default_rng(0)
        # window rates (non-overlapping 1024, the judge's window) vs iid
        nw = n // W
        win = wrong_let[:nw * W].reshape(nw, W).mean(1)
        iid = np.stack([rng.permutation(wrong_let)[:nw * W].reshape(nw, W).mean(1) for _ in range(200)])
        # runs of correct letters
        cr = runs_of(~wrong_let); cr_iid = runs_of(~rng.permutation(wrong_let))
        # lag-1 autocorrelation of the error indicator at letter and token level
        def ac(x): x = x - x.mean(); return float((x[1:] * x[:-1]).mean() / x.var())
        # a "clean 100-char stretch": fraction of 100-char windows with <=2 errors
        w100 = wrong_let[:(n // 100) * 100].reshape(-1, 100).sum(1)
        w100_iid = rng.permutation(wrong_let)[:(n // 100) * 100].reshape(-1, 100).sum(1)
        rec = dict(cell=name, seed=r["seed"], ser=round(float(ser), 3), letter_err=round(float(p), 3),
                   n_letters=int(n), n_tokens=int(len(sym)),
                   wrong_types=int((m != true_map).sum()), n_types=int(len(m)),
                   wrong_types_occ_ge10=int(((m != true_map) & (occ >= 10)).sum()),
                   share_of_errors_from_types_occ_ge10=round(float((occ * (m != true_map))[occ >= 10].sum() / (occ * (m != true_map)).sum()), 3),
                   win1024_rates=np.round(win, 3).tolist(),
                   win1024_sd=round(float(win.std()), 4), win1024_sd_iid=round(float(iid.std(1).mean()), 4),
                   win1024_min=round(float(win.min()), 3), win1024_min_iid=round(float(iid.min(1).mean()), 3),
                   win1024_max=round(float(win.max()), 3), win1024_max_iid=round(float(iid.max(1).mean()), 3),
                   longest_correct_run=int(cr.max()), longest_correct_run_iid=int(cr_iid.max()),
                   mean_correct_run=round(float(cr.mean()), 2), mean_correct_run_iid=round(float(cr_iid.mean()), 2),
                   p_correct_run_ge20=round(float((cr >= 20).mean()), 4), p_correct_run_ge20_iid=round(float((cr_iid >= 20).mean()), 4),
                   frac100_le2err=round(float((w100 <= 2).mean()), 3), frac100_le2err_iid=round(float((w100_iid <= 2).mean()), 3),
                   frac100_ge20err=round(float((w100 >= 20).mean()), 3), frac100_ge20err_iid=round(float((w100_iid >= 20).mean()), 3),
                   ac1_letter=round(ac(wrong_let.astype(float)), 3), ac1_token=round(ac(wrong_tok.astype(float)), 3))
        out.append(rec)
        print(json.dumps(rec))
(root / "analysis/altloop/error_spread.json").write_text(json.dumps(out, indent=1))
