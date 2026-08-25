"""Fetch and pin all external artifacts (tasks 0.7, 0.8; cross-cutting X.4).

Idempotent: clones/downloads only what is missing, then force-checks-out the
pinned SHAs from ``diff_voyn.ciphers.external``. Everything lands under
DATA_ROOT (default /workspace/data), the host bind mount.

Run: ``uv run python scripts/fetch_external.py``
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diff_voyn.ciphers.external import (
    CIPHER_BENCHMARK_SHA,
    CIPHER_BENCHMARK_SPARSE,
    CIPHER_BENCHMARK_URL,
    GUTENBERG_ENGLISH_IDS,
    NAIBBE_SHA,
    NAIBBE_URL,
    VOYNICH_ATTACK_SHA,
    VOYNICH_ATTACK_URL,
    data_root,
)

VOYNICH_NU_FILES = ["IT2a-n.txt", "RF1b-e.txt", "000_README.txt"]


def clone_and_pin(url: str, dest: Path, sha: str) -> None:
    if not dest.is_dir():
        print(f"cloning {url} -> {dest}")
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != sha:
        print(f"pinning {dest.name} to {sha[:12]}")
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet"], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)


def sparse_clone_and_pin(url: str, dest: Path, sha: str, paths) -> None:
    """Blob-less sparse clone (text directories only) pinned to ``sha``."""
    if not dest.is_dir():
        print(f"sparse-cloning {url} -> {dest}")
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--sparse",
                url,
                str(dest),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "sparse-checkout", "set", *paths], check=True
        )
    clone_and_pin(url, dest, sha)


def main() -> None:
    root = data_root()
    ext = root / "external"
    ext.mkdir(parents=True, exist_ok=True)
    clone_and_pin(NAIBBE_URL, ext / "naibbe-cipher", NAIBBE_SHA)
    clone_and_pin(VOYNICH_ATTACK_URL, ext / "voynich-attack", VOYNICH_ATTACK_SHA)

    vms_dir = root / "raw" / "vms"
    vms_dir.mkdir(parents=True, exist_ok=True)
    for name in VOYNICH_NU_FILES:
        target = vms_dir / name
        if not target.exists():
            print(f"downloading voynich.nu/data/{name}")
            urllib.request.urlretrieve(f"http://www.voynich.nu/data/{name}", target)
    # Phase 6 anchors (task 6.6)
    sparse_clone_and_pin(
        CIPHER_BENCHMARK_URL,
        ext / "cipher_benchmark",
        CIPHER_BENCHMARK_SHA,
        CIPHER_BENCHMARK_SPARSE,
    )
    eng = root / "raw" / "anchors" / "english"
    eng.mkdir(parents=True, exist_ok=True)
    for gid in GUTENBERG_ENGLISH_IDS:
        target = eng / f"pg{gid}.txt"
        if not target.exists():
            print(f"downloading gutenberg pg{gid}.txt")
            urllib.request.urlretrieve(
                f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", target
            )
    print("external artifacts ready")


if __name__ == "__main__":
    main()
