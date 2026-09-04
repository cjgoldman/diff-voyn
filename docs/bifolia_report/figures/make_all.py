"""Regenerate every report figure: runs each fig_*.py in this directory.

    cd docs/bifolia_report/figures && uv run python make_all.py [pattern]

An optional substring argument restricts the run to matching scripts.
"""

import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
pattern = sys.argv[1] if len(sys.argv) > 1 else ""
scripts = sorted(p for p in glob.glob(os.path.join(HERE, "fig_*.py")) if pattern in os.path.basename(p))
failed = []
for s in scripts:
    name = os.path.basename(s)
    print(f"-- {name}", flush=True)
    r = subprocess.run(["uv", "run", "python", s], cwd=HERE, capture_output=True, text=True)
    err = "\n".join(l for l in r.stderr.splitlines() if "findfont" not in l)
    if r.returncode != 0:
        failed.append(name)
        print(r.stdout, err)
    elif err.strip():
        print(err)
print(f"{len(scripts) - len(failed)}/{len(scripts)} figures built" + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
