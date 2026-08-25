"""Pinned external cipher repositories (task 0.7, X.4).

The repos live under ``DATA_ROOT/external`` (cloned by
``scripts/fetch_external.py``); every import verifies the checked-out commit
against the pin recorded here, so silent drift is impossible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Pins. Naibbe pin comes from the design doc (§10); voynich-attack is pinned
# to the "latest" at acquisition time per task 0.7.
NAIBBE_SHA = "df3d07402643ca27e8a867e615e94f076d9215bc"
VOYNICH_ATTACK_SHA = "e324beec4312483551554bc3396515286f35336d"

NAIBBE_URL = "https://github.com/greshko/naibbe-cipher.git"
VOYNICH_ATTACK_URL = "https://github.com/alexanderboxer/voynich-attack.git"

# Phase 6 (task 6.6): benchmark anchors — the Borg cipher transcription +
# Latin plaintext and the Zodiac-408 record, from matthewdgreen/
# cipher_benchmark (sparse clone, text only; pinned at acquisition 2026-08-23).
CIPHER_BENCHMARK_SHA = "729aad62d12483c549e64a2541d4f9255538c8cf"
CIPHER_BENCHMARK_URL = "https://github.com/matthewdgreen/cipher_benchmark.git"
CIPHER_BENCHMARK_SPARSE = (
    "benchmark/manifest",
    "benchmark/docs",
    "benchmark/sources/borg/transcriptions",
    "benchmark/sources/borg/plaintext",
    "benchmark/sources/borg/metadata",
    "benchmark/sources/zodiac",
    "scripts",
)
# Modern-English public-domain texts for the Zodiac-408 n-gram baseline
# (English is outside the frozen inventory). Project Gutenberg ids.
GUTENBERG_ENGLISH_IDS = (98, 1342, 2701, 1661, 84, 11, 345, 76)


def data_root() -> Path:
    return Path(os.environ.get("DIFF_VOYN_DATA", "/workspace/data"))


def _verify(repo: Path, expected_sha: str) -> Path:
    if not repo.is_dir():
        raise FileNotFoundError(
            f"{repo} missing — run `uv run python scripts/fetch_external.py`"
        )
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if sha != expected_sha:
        raise RuntimeError(
            f"{repo} is at {sha[:12]}, expected pin {expected_sha[:12]} "
            "(task 0.7 / X.4: re-run scripts/fetch_external.py)"
        )
    return repo


def naibbe_repo() -> Path:
    return _verify(data_root() / "external" / "naibbe-cipher", NAIBBE_SHA)


def cipher_benchmark_repo() -> Path:
    return _verify(data_root() / "external" / "cipher_benchmark", CIPHER_BENCHMARK_SHA)


def voynich_attack_repo() -> Path:
    return _verify(data_root() / "external" / "voynich-attack", VOYNICH_ATTACK_SHA)


def ensure_voynpy_importable() -> None:
    repo = str(voynich_attack_repo())
    if repo not in sys.path:
        sys.path.insert(0, repo)
