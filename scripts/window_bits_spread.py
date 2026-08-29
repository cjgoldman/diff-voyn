"""Per-window judge bits vs per-window error rate for the A-like anneal finals."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from diff_voyn.ciphers.external import data_root
from diff_voyn.heads.two_tier import paired_bits
from diff_voyn.heads.wordhom import UnitTargets, expand_units
from diff_voyn.heads.diffusion_eval import DiffusionEvaluator
from diff_voyn.metrology import CALIBRATION_VERSION, CalibrationTable, calibrate_bits
from diff_voyn.vms.apply import LANGS
torch.set_float32_matmul_precision("high")
root = data_root(); W = 1024
CELLS = [("positive/german/Alike", "german", "runs_anneal_de.json"),
         ("positive/italian/Alike", "italian", "runs_anneal.json"),
         ("positive/latin/Alike", "latin", "runs_anneal.json")]
ev = DiffusionEvaluator.from_checkpoint(root / "runs/phase_c-85m-seed0/ckpt_final.pt", device="cuda")
offs = CalibrationTable.load(CALIBRATION_VERSION, root).additive_offsets()
out = []
for name, lang, fn in CELLS:
    inst = json.loads((root / "analysis/wordhom/controls/wordtypesall" / (name.replace("/", "_") + "_wordtypesall.json")).read_text())
    tr = inst["truth"]; true_map = np.asarray(tr["sym_to_unit"]); targets = UnitTargets.from_list(tr["bigrams"])
    sym = np.asarray(inst["symbols"])
    keys = [("truth", true_map)] + [(f"anneal/s{r['seed']}", np.asarray(r["final_map"])) for r in json.loads((root / "analysis/altloop" / fn).read_text()) if r["cell"] == f"wh/{name}/{lang}"]
    for kname, m in keys:
        dec = expand_units(m[sym], targets)
        wrong = np.repeat((m != true_map)[sym], 1 + (targets.second[m[sym]] >= 0))
        rng = np.random.default_rng(0)
        for wi, s in enumerate(range(0, len(dec) - W + 1, W)):
            d = dec[s:s+W]; rows = np.stack([d, rng.permutation(d)])
            vals = []
            for sd in range(2):
                pb = paired_bits(ev, rows, list(LANGS), n_strata=64, seed=1000*sd + 17*wi)
                j = LANGS.index(lang)
                vals.append((calibrate_bits(float(pb[0, j]), lang, offs), calibrate_bits(float(pb[1, j]), lang, offs)))
            dbits = float(np.mean([v[0] for v in vals])); sbits = float(np.mean([v[1] for v in vals]))
            rec = dict(cell=name, key=kname, win=wi, err=float(wrong[s:s+W].mean()), decode_bits=dbits, shuffled_bits=sbits, margin=sbits - dbits)
            out.append(rec); print(json.dumps(rec), flush=True)
(root / "analysis/altloop/window_bits_spread.json").write_text(json.dumps(out, indent=1))
